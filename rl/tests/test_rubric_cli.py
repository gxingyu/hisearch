from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from grepseek.rubric.client import RubricClient
from grepseek.rubric.core import RubricError, load_snapshot
from rl import rubric_workflow as cli
from rl.tests.test_rubric_workflow import INITIAL_PATH, JudgeServer, trajectory


class RubricCliTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.initial=load_snapshot(INITIAL_PATH)
        self.server=JudgeServer()
        self.client=RubricClient({'base_url':'http://rubric.invalid/v1','model':'mock','retries':0},
                                 transport=httpx.MockTransport(self.server))
        self.env=patch.dict(os.environ,{'GREPSEEK_RUBRIC_API_KEY':'test-only-token'})
        self.env.start()

    def tearDown(self):self.env.stop();self.temp.cleanup()

    def args(self,**kw):
        return SimpleNamespace(initial=str(INITIAL_PATH),output=str(self.root/'output.json'),
            input=str(self.root/'input.jsonl'),train_file=str(self.root/'train.jsonl'),
            question_limit=16,model='mock',calibration='',require_calibration=False,
            live=False,limit=16,**kw)

    def write_rows(self,path,rows):Path(path).write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_check_only_is_lightweight_and_never_imports_training_workers(self):
        repository=Path(__file__).resolve().parents[2]
        program="import sys; from rl import rubric_workflow as c; sys.argv=['rubric','check']; c.main(); assert 'torch' not in sys.modules; assert 'ray' not in sys.modules; assert 'vllm' not in sys.modules"
        result=subprocess.run([sys.executable,'-c',program],cwd=repository,env=os.environ.copy(),
                              capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        report=json.loads(result.stdout)
        self.assertFalse(report['policy_training_started'])
        self.assertFalse(report['network_checked'])
        self.assertFalse(report['dynamic_publication_ready'])

    def test_check_requires_real_reference_coverage_before_explicit_train(self):
        args=self.args();args.require_calibration=True
        with self.assertRaisesRegex(RubricError,'reference panel'):cli.check(args)

    def test_initialize_writes_model_provenance_and_preserves_five_ids(self):
        args=self.args()
        self.write_rows(args.train_file,[{'id':f'train-{i}','question':f'Question {i}?'} for i in range(20)])
        def generate(request):
            data=json.loads(json.loads(request.content)['messages'][1]['content'])
            self.assertEqual(len(data['training_questions']),16)
            body={'rubrics':data['template']['rubrics']}
            return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(body)},'finish_reason':'stop'}]})
        client=RubricClient({'base_url':'http://rubric.invalid/v1','model':'mock'},transport=httpx.MockTransport(generate))
        with patch.object(cli,'client_for',return_value=client),contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(cli.initialize(args))
        snapshot=load_snapshot(args.output)
        self.assertTrue(snapshot['origin'].startswith('offline_model:mock;'))
        self.assertEqual(len(snapshot['rubrics']),5)
        self.assertTrue(Path(args.output+'.audit.json').is_file())
        with self.assertRaisesRegex(RubricError,'overwrite'):asyncio.run(cli.initialize(args))

    def test_prepare_reference_records_model_labels_and_initial_hash(self):
        args=self.args()
        rows=[{**trajectory(f'ref-{i}',passed=i%2==0),'split':'train'} for i in range(8)]
        self.write_rows(args.input,rows)
        with patch.object(cli,'client_for',return_value=self.client),contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(cli.prepare_calibration(args))
        references=cli.read_jsonl(args.output)
        self.assertEqual(len(references),8)
        self.assertTrue(all('not_human_labels' in r['reference_source'] for r in references))
        args.calibration=args.output;args.require_calibration=True
        with contextlib.redirect_stdout(io.StringIO()) as output:cli.check(args)
        self.assertTrue(json.loads(output.getvalue())['dynamic_publication_ready'])

    def test_reference_preparation_rejects_validation_rows(self):
        args=self.args();self.write_rows(args.input,[{**trajectory('validation'),'split':'val'}])
        with self.assertRaisesRegex(RubricError,'training-only'):asyncio.run(cli.prepare_calibration(args))

    def test_export_reads_only_archived_visible_text_and_joins_train_qids(self):
        args=self.args()
        self.write_rows(args.train_file,[{'id':'train-q','question':'Which relation?'}])
        output='<think>Find evidence.</think><tool_call>{"query":"relation"}</tool_call><tool_response>Only shown text.</tool_response><answer>A</answer>'
        self.write_rows(args.input,[{'gts':{'qid':'train-q'},'output':output,'step':40,
                                    'reward/format_pass':1,'reward/f1':1},
                                   {'gts':{'qid':'not-in-train'},'output':output,'reward/format_pass':1,'reward/f1':1}])
        with contextlib.redirect_stdout(io.StringIO()):cli.export_inputs(args)
        rows=cli.read_jsonl(args.output)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['source_step'],40)
        self.assertEqual(''.join(e['text'] for e in rows[0]['events'][1:]),output)
        self.assertEqual(rows[0]['split'],'train')
        self.assertEqual(rows[0]['step'],1)

    def test_export_rejects_validation_path(self):
        args=self.args();args.input=str(self.root/'val_rollouts/0.jsonl')
        with self.assertRaisesRegex(RubricError,'validation'):cli.export_inputs(args)

    def test_replay_scores_and_saves_steps_without_actor_or_optimizer(self):
        args=self.args();args.output=str(self.root/'replay')
        rows=[{**trajectory(f'q-{step}',number=i,passed=i==0),'split':'train','step':step}
              for step in (1,2) for i in range(2)]
        self.write_rows(args.input,rows)
        with patch.object(cli,'client_for',return_value=self.client),contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(cli.replay(args))
        report=json.loads((Path(args.output)/'replay_report.json').read_text())
        self.assertFalse(report['policy_training_started'])
        self.assertEqual(len(report['steps']),2)
        self.assertEqual(report['steps'][0]['statistics']['trajectories'],2)
        self.assertEqual(report['steps'][0]['decision']['action'],'KEEP')


if __name__=='__main__':unittest.main()
