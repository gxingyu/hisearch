# Vendored verl runtime

This is the complete Python runtime from the research checkout's vendored
[verl](https://github.com/volcengine/verl), including local HiSearch training
fixes. It is included as source, not a submodule. See [VENDORED.md](VENDORED.md),
[LICENSE](LICENSE), and [Notice.txt](Notice.txt).

Launchers load this directory through `PYTHONPATH`. Follow
[the policy installation recipe](../TRAINING_ENV.md) for the actual dependency
versions; upstream packaging extras do not describe this local stack.
