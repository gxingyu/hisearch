from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from grepseek.rubric.core import (
    RubricError, RubricWorkflow, combine_group_rewards, digest, load_snapshot,
    summarize, validate_grades, validate_snapshot, visible_events,
)
from grepseek.rubric.client import RubricClient, GRADE_PROMPT, MAINTAIN_PROMPT, REVIEW_PROMPT

INITIAL_PATH = Path(__file__).resolve().parents[1] / 'grepseek/trainer/config/rubric/initial_rubrics.json'


def trajectory(qid, group=None, number=0, *, passed=True, f1=.6, valid=True):
    answer = 'supported answer' if passed else 'unsupported claim'
    return {'trajectory_id': f'{qid}:{number}', 'group_id': group or qid, 'qid': qid,
            'question': 'Which relation is established?', 'valid': valid, 'f1': f1,
            'base_score': f1 if valid else -1., 'events': [
                {'id': 'question', 'kind': 'question', 'text': 'Which relation is established?'},
                {'id': 'action_1', 'kind': 'action', 'text': 'search for the relation'},
                {'id': 'obs_2', 'kind': 'observation', 'text': 'Visible evidence for the relation.'},
                {'id': 'answer', 'kind': 'answer', 'text': answer}], 'buckets': ['corpus']}


def response_grades(snapshot, record, statuses=None):
    status = 'pass' if record['events'][-1]['text'] == 'supported answer' else 'fail'
    return {'grades': [{'rubric_id': item['id'], 'status': (statuses or {}).get(item['id'], status),
                        'evidence': [{'event_id': 'answer', 'quote': record['events'][-1]['text']}],
                        'reason': 'Check the actual visible relation.'} for item in snapshot['rubrics']]}


class JudgeServer:
    """In-process httpx transport. No socket, process, GPU, or external request."""
    def __init__(self):
        self.requests = []
        self.invalid_quotes = False
        self.unavailable = False
        self.reject_review = False
        self.reject_calibration = False
        self.uncertain_id = None
        self.proposal_action = 'REVISE'
        self.proposal_transform = lambda value: value

    def __call__(self, request):
        payload = json.loads(request.content)
        data = json.loads(payload['messages'][1]['content'])
        system = payload['messages'][0]['content']
        self.requests.append((system, data))
        if self.unavailable:
            return httpx.Response(503, json={'error': 'temporarily unavailable'})
        if system == GRADE_PROMPT:
            # Grader cannot observe answer labels, rewards, or unseen metadata.
            assert set(data['trajectory']) == {'trajectory_id', 'question', 'events'}
            snapshot, record = data['rubric_snapshot'], data['trajectory']
            overrides = {}
            if self.uncertain_id:
                overrides[self.uncertain_id] = 'uncertain'
            if self.reject_calibration and snapshot['version'] > 0 and record['trajectory_id'].startswith('cal-'):
                overrides[snapshot['rubrics'][0]['id']] = 'fail'
            result = response_grades(snapshot, record, overrides)
            if self.invalid_quotes:
                result['grades'][0]['evidence'][0]['quote'] = 'UNSEEN BACKEND CONTENT'
        elif system == MAINTAIN_PROMPT:
            if self.proposal_action == 'KEEP':
                result = {'action': 'KEEP', 'reason': 'The failure reflects a genuine capability gap.'}
            else:
                identifier = data['eligible'][0]['rubric_id']
                item = copy.deepcopy(next(r for r in data['snapshot']['rubrics'] if r['id'] == identifier))
                item['revision'] += 1
                item['criterion'] += ' Judge using the visible relation rather than answer-string occurrence.'
                result = self.proposal_transform({'action': 'REVISE', 'rubric_id': identifier, 'item': item,
                          'reason': 'Clarify a saturated predicate without changing its goal.',
                          'case_ids': [data['examples'][0]['trajectory_id']]})
        elif system == REVIEW_PROMPT:
            result = {'preserves_core': not self.reject_review, 'tool_neutral': True,
                      'answer_free': True, 'reason': 'Review of the original process obligation.'}
        else:
            raise AssertionError('Unexpected model operation')
        return httpx.Response(200, json={'choices': [{'message': {'content': json.dumps(result)}, 'finish_reason': 'stop'}],
                                        'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})


class RubricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.initial = load_snapshot(INITIAL_PATH)
        self.server = JudgeServer()
        self.environment = patch.dict(os.environ, {'GREPSEEK_RUBRIC_API_KEY': 'test-only-token'})
        self.environment.start()
        self.client = RubricClient({'base_url': 'http://rubric.invalid/v1', 'model': 'mock-27b',
            'concurrency': 3, 'retries': 1, 'audit_path': str(self.root/'requests.jsonl')},
            transport=httpx.MockTransport(self.server))
        self.panel = self.root/'reference.jsonl'
        references = []
        for qid, passed in [('cal-positive', True), ('cal-negative', False)]:
            r = trajectory(qid, passed=passed)
            r.update(split='train', initial_snapshot_hash=digest(self.initial), reference_source='synthetic_test_fixture',
                     reference_statuses={item['id']: 'pass' if passed else 'fail' for item in self.initial['rubrics']})
            references.append(r)
        self.panel.write_text(''.join(json.dumps(r)+'\n' for r in references))

    def tearDown(self):
        self.environment.stop()
        self.temp.cleanup()

    def workflow(self, **overrides):
        options = dict(min_clear=4, min_questions=2, consecutive_checks=3, min_calibration_questions=2,
                       max_calibration_examples=2, max_examples=8)
        options.update(overrides)
        return RubricWorkflow(self.initial, self.root/'workflow', self.client,
                              calibration_path=self.panel, maintenance=options)

    def rows(self, step, passed=True):
        return [trajectory(f'train-{step}-{q}', number=j, passed=passed) for q in range(2) for j in range(2)]

    def test_reward_is_f1_plus_point_one_without_clipping_and_invalid_is_unchanged(self):
        a, b, invalid = trajectory('q', number=0), trajectory('q', number=1, passed=False), trajectory('q', number=2, valid=False)
        grades = [validate_grades(response_grades(self.initial, r), self.initial, r) for r in (a,b)] + [None]
        results = combine_group_rewards([a,b,invalid], grades)
        self.assertAlmostEqual(results[0]['score'], 1.1)
        self.assertAlmostEqual(results[1]['score'], .6)
        self.assertEqual(results[2]['score'], -1.)
        self.assertEqual(results[2]['bonus'], 0)
        a['base_score'] = a['f1'] = 1.
        self.assertAlmostEqual(combine_group_rewards([a],[grades[0]])[0]['score'], 1.5)

    def test_uncertain_masks_one_item_for_the_group_and_errors_fallback_whole_group(self):
        rows = [trajectory('q', number=i) for i in range(2)]
        grades = [validate_grades(response_grades(self.initial,r),self.initial,r) for r in rows]
        identifier = self.initial['rubrics'][0]['id']
        grades[1][identifier]['status'] = 'uncertain'
        results = combine_group_rewards(rows, grades)
        self.assertEqual([r['passed'] for r in results], [4,4])
        self.assertEqual([r['masked_items'] for r in results], [1,1])
        results = combine_group_rewards(rows, [grades[0],None])
        self.assertTrue(all(r['fallback'] and r['score']==.6 for r in results))

    def test_fabricated_evidence_retries_and_never_gets_a_pass(self):
        self.server.invalid_quotes = True
        result = asyncio.run(self.client.grade_many(self.initial,[trajectory('q')]))
        self.assertEqual(result,[None])
        self.assertEqual(self.client.usage['requests'],2)
        audit=(self.root/'requests.jsonl').read_text()
        self.assertNotIn('test-only-token',audit)
        self.assertIn('verbatim',audit)

    def test_http_failure_is_not_behavioral_failure(self):
        self.server.unavailable=True
        flow=self.workflow()
        flow.begin_step(1)
        rows=asyncio.run(flow.score_groups(self.rows(1)))
        self.assertTrue(all(r['fallback'] for r in rows))
        stats,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in rows]))
        self.assertEqual(stats['judge_errors'],4)
        self.assertTrue(all(s['clear']==0 for s in stats['items'].values()))
        self.assertEqual(decision['action'],'KEEP')

    def test_one_http_call_grades_all_five_and_gold_is_never_sent(self):
        row=trajectory('q')
        row['golden_answers']=['secret-reference-answer']
        row['events'][0]['hidden_backend_text']='nested-secret-reference'
        result=asyncio.run(self.client.grade_many(self.initial,[row]))
        self.assertEqual(len(result[0]),5)
        self.assertEqual(self.client.usage['requests'],1)
        serialized=json.dumps(self.server.requests)
        self.assertNotIn('secret-reference-answer',serialized)
        self.assertNotIn('base_score',serialized)
        self.assertNotIn('"f1"',serialized)
        self.assertNotIn('nested-secret-reference',serialized)

    def test_same_global_pass_rate_can_have_zero_or_full_group_discrimination(self):
        def scored(rows):
            return [{**r,'grades':validate_grades(response_grades(self.initial,r),self.initial,r)} for r in rows]
        a=scored([trajectory('a',number=i) for i in range(5)]+[trajectory('b',number=i,passed=False) for i in range(5)])
        b=scored([trajectory('a',number=i,passed=i<3) for i in range(5)]+[trajectory('b',number=i,passed=i<2) for i in range(5)])
        identifier=self.initial['rubrics'][0]['id']
        sa,sb=(summarize(r,self.initial)['items'][identifier] for r in (a,b))
        self.assertEqual(sa['pass_rate'],sb['pass_rate'])
        self.assertEqual((sa['group_discrimination'],sb['group_discrimination']),(0.,1.))

    def test_question_balancing_and_uncertainty_denominators(self):
        rows=[trajectory('a',number=i) for i in range(5)]+[trajectory('b',passed=False)]
        for r in rows:r['grades']=validate_grades(response_grades(self.initial,r),self.initial,r)
        identifier=self.initial['rubrics'][0]['id']
        stats=summarize(rows,self.initial)['items'][identifier]
        self.assertEqual(stats['raw_pass_rate'],5/6)
        self.assertEqual(stats['pass_rate'],.5)
        rows[0]['grades'][identifier]['status']='uncertain'
        rows[1]['grades'][identifier]['status']='not_applicable'
        stats=summarize(rows,self.initial)['items'][identifier]
        self.assertEqual(stats['clear'],4)
        self.assertEqual(stats['uncertainty_rate'],1/5)

    def test_three_mature_checks_publish_only_next_step_and_checkpoint_resumes(self):
        flow=self.workflow()
        for step in (1,2,3):
            frozen=flow.begin_step(step)
            # Two refill calls in the same outer step retain the same version.
            rows=self.rows(step)
            a=asyncio.run(flow.score_groups(rows[:2]))
            b=asyncio.run(flow.score_groups(rows[2:]))
            self.assertTrue(all(r['rubric_version']==0 for r in a+b))
            stats,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in a]))
            self.assertEqual(stats['trajectories'],4)
            self.assertEqual(stats['retained_trajectories'],2)
            self.assertEqual(decision['action'],'REVISE' if step==3 else 'KEEP')
            self.assertEqual(frozen['version'],0)
        self.assertEqual(flow.active['version'],1)
        self.assertEqual(flow.active['effective_from_step'],4)
        self.assertEqual(sum(r['revision'] for r in flow.active['rubrics']),1)
        state=flow.state_dict()
        restored=self.workflow()
        restored.restore(state,3)
        self.assertEqual(restored.begin_step(4)['version'],1)
        with self.assertRaises(RubricError):self.workflow().restore(state,2)
        changed=copy.deepcopy(state);changed['judge_signature']['model']='different-judge'
        with self.assertRaisesRegex(RubricError,'judge model'):self.workflow().restore(changed,3)

    def test_semantic_rejection_keeps_old_snapshot(self):
        self.server.reject_review=True
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        scored=asyncio.run(flow.score_groups(self.rows(1)))
        _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        self.assertEqual(decision['reason'],'semantic_review_rejected')
        self.assertEqual(flow.active,self.initial)

    def test_false_penalty_reference_rejection_keeps_old_snapshot(self):
        self.server.reject_calibration=True
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        scored=asyncio.run(flow.score_groups(self.rows(1)))
        _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        self.assertEqual(decision['reason'],'reference_panel_rejected')
        self.assertEqual(flow.active['version'],0)

    def test_reference_qids_are_excluded_from_proposal_examples(self):
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        rows=self.rows(1)+[trajectory('cal-positive',number=i+10) for i in range(2)]
        scored=asyncio.run(flow.score_groups(rows))
        asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        requests=[data for system,data in self.server.requests if system==MAINTAIN_PROMPT]
        self.assertEqual(len(requests),1)
        self.assertTrue(all(not e['trajectory_id'].startswith('cal-') for e in requests[0]['examples']))

    def test_unavailable_or_sparse_calibration_fails_closed(self):
        flow=RubricWorkflow(self.initial,self.root/'no-panel',self.client,
             maintenance={'min_clear':4,'min_questions':2,'consecutive_checks':1,'min_calibration_questions':2})
        flow.begin_step(1)
        scored=asyncio.run(flow.score_groups(self.rows(1)))
        _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        self.assertEqual(decision['reason'],'missing_training_reference_panel')
        self.assertEqual(flow.active['version'],0)

    def test_high_failure_is_diagnosed_without_deleting_target(self):
        self.server.proposal_action='KEEP'
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        rows=asyncio.run(flow.score_groups(self.rows(1,passed=False)))
        _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in rows]))
        request=next(data for system,data in self.server.requests if system==MAINTAIN_PROMPT)
        self.assertEqual(request['eligible'][0]['reason'],'diagnose_failure_not_remove_target')
        self.assertEqual(decision['action'],'KEEP')
        self.assertEqual(flow.active,self.initial)

    def test_all_uncertain_can_trigger_clarification_without_clear_labels(self):
        self.server.uncertain_id=self.initial['rubrics'][0]['id']
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        rows=asyncio.run(flow.score_groups(self.rows(1)))
        stats,_=asyncio.run(flow.finish_step([r['trajectory_id'] for r in rows]))
        self.assertEqual(stats['items'][self.server.uncertain_id]['clear'],0)
        request=next(data for system,data in self.server.requests if system==MAINTAIN_PROMPT)
        self.assertEqual(request['eligible'][0]['reason'],'ambiguous')

    def test_sparse_steps_accumulate_but_changed_revision_drops_old_evidence(self):
        flow=self.workflow(min_clear=8,min_questions=4,consecutive_checks=1)
        for step in (1,2):
            flow.begin_step(step)
            rows=asyncio.run(flow.score_groups(self.rows(step)))
            _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in rows]))
            self.assertEqual(decision['action'],'KEEP' if step==1 else 'REVISE')
        from grepseek.rubric.core import window_statistics
        stats=window_statistics(flow.history,flow.active)
        changed=flow.active['rubrics'][0]['id']
        self.assertEqual(stats['items'][changed]['clear'],0)
        self.assertEqual(stats['items'][self.initial['rubrics'][1]['id']]['clear'],8)
        restored=self.workflow(min_clear=8,min_questions=4,consecutive_checks=1)
        restored.restore(flow.state_dict(),2)
        self.assertEqual(restored.history,flow.history)

    def test_window_does_not_count_repeated_question_ids_as_new_questions(self):
        self.server.proposal_action='KEEP'
        flow=self.workflow(min_clear=8,min_questions=3,consecutive_checks=1)
        for step in (1,2):
            flow.begin_step(step)
            rows=self.rows(1)
            for r in rows:
                r['group_id']+=f':step{step}'
                r['trajectory_id']+=f':step{step}'
            scored=asyncio.run(flow.score_groups(rows))
            asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        self.assertFalse(any(system==MAINTAIN_PROMPT for system,data in self.server.requests))

    def test_aborted_step_does_not_advance_evidence_or_block_retry(self):
        flow=self.workflow()
        flow.begin_step(1)
        asyncio.run(flow.score_groups(self.rows(1)))
        flow.abort_step('partial rollout never reached an actor update')
        self.assertEqual(flow.completed_step,0)
        self.assertEqual(flow.history,[])
        self.assertEqual(flow.begin_step(1)['version'],0)
        self.assertEqual(len(list((self.root/'workflow/steps').glob('*/aborted.json'))),1)

    def test_proposal_examples_preserve_same_question_contrasts(self):
        flow=self.workflow(consecutive_checks=1)
        flow.begin_step(1)
        # One item is saturated; other grades still differ within qids.
        rows=self.rows(1)+[trajectory('mixed',number=i,passed=i==0) for i in range(2)]
        scored=asyncio.run(flow.score_groups(rows))
        scored_id=self.initial['rubrics'][0]['id']
        for record in flow.records:
            record['grades'][scored_id]['status']='pass'
        asyncio.run(flow.finish_step([r['trajectory_id'] for r in scored]))
        request=next(data for system,data in self.server.requests if system==MAINTAIN_PROMPT)
        self.assertEqual(sum(e['trajectory_id'].startswith('mixed:') for e in request['examples']),2)

    def test_foreign_case_or_weight_change_is_rejected(self):
        for mutation in ('case','weight','revision','nested_case','invalid_id','noop'):
            with self.subTest(mutation=mutation):
                def change(proposal):
                    if mutation=='case':proposal['case_ids']=['unseen-qid']
                    if mutation=='weight':proposal['item']['weight']=.2
                    if mutation=='revision':proposal['item']['revision']=22
                    if mutation=='nested_case':proposal['case_ids']=[{'invalid':'case'}]
                    if mutation=='invalid_id':proposal['rubric_id']=['not-a-string']
                    if mutation=='noop':proposal['item']['criterion']=self.initial['rubrics'][0]['criterion']
                    return proposal
                self.server.proposal_transform=change
                self.client.memory.clear()
                flow=self.workflow(consecutive_checks=1)
                flow.begin_step(1)
                rows=asyncio.run(flow.score_groups(self.rows(1)))
                _,decision=asyncio.run(flow.finish_step([r['trajectory_id'] for r in rows]))
                self.assertEqual(decision['action'],'KEEP')
                self.assertEqual(flow.active['version'],0)

    def test_version_bound_cache_and_duplicate_group_guard(self):
        row=trajectory('q')
        asyncio.run(self.client.grade_many(self.initial,[row]))
        asyncio.run(self.client.grade_many(self.initial,[row]))
        self.assertEqual(self.client.usage['requests'],1)
        changed=copy.deepcopy(self.initial)
        changed['version']=1
        changed['effective_from_step']=2
        asyncio.run(self.client.grade_many(changed,[row]))
        self.assertEqual(self.client.usage['requests'],2)
        flow=self.workflow()
        flow.begin_step(1)
        asyncio.run(flow.score_groups([row]))
        with self.assertRaises(RubricError):asyncio.run(flow.score_groups([row]))

    def test_snapshot_requires_exactly_five_and_no_weights(self):
        bad=copy.deepcopy(self.initial);bad['rubrics'].pop()
        with self.assertRaises(RubricError):validate_snapshot(bad)
        bad=copy.deepcopy(self.initial);bad['rubrics'][0]['weight']=.1
        with self.assertRaises(RubricError):validate_snapshot(bad)

    def test_visible_ledger_uses_only_actual_token_spans_and_keeps_prefills(self):
        class Tokenizer:
            @staticmethod
            def decode(ids,**kwargs):return ''.join(map(chr,ids))
        chunks=['search','shown truncated result','<answer>','supported answer']
        ids=[ord(c) for chunk in chunks for c in chunk]
        mask=[1]*len(chunks[0])+[0]*(len(chunks[1])+len(chunks[2]))+[1]*len(chunks[3])
        events=visible_events(Tokenizer(),ids,mask)
        self.assertEqual([e['kind'] for e in events],['action','observation','answer'])
        self.assertEqual(''.join(e['text'] for e in events),''.join(chunks))
        self.assertNotIn('UNSEEN',json.dumps(events))
        with self.assertRaises(RubricError):visible_events(Tokenizer(),ids,mask[:-1])


if __name__=='__main__':unittest.main()
