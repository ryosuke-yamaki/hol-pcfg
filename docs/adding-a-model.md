# Adding a model

Models are selected purely by the `model_name` string in a config's `model:` block.
Wiring a new model takes three code touchpoints plus a config.

1. **Define the model** — add `parser/model/<Your>_PCFG.py` with an `nn.Module`
   subclass exposing the same interface as the existing models (e.g. `HN_PCFG`):
   `forward(...)` returning a dict of rule-probability tensors, a `loss(...)` returning
   the scalar NLL (this is what the training loop calls — see `parser/cmds/cmd.py`), and
   an `evaluate(...)` used by the eval loop.
   Reuse the inside/decode backends in `parser/pcfgs/` (e.g. `SimplePCFG_Triton`,
   `SimplePCFG_Triton_Batch`) rather than re-implementing CKY.

2. **Export it** — add the class to `parser/model/__init__.py`.

3. **Register the dispatch** — add an `elif args.model_name == "<NAME>":` branch in
   `parser/helper/util.py:get_model` returning your class. `get_model` raises
   `KeyError` for unknown names, so an unregistered `model_name` fails fast.

4. **Add a config** — create a YAML under the appropriate `config/<family>/` dir with
   `model.model_name: "<NAME>"` and the schema documented in the top-level `README.md`.

Run it with:

```bash
python train.py --conf config/<family>/<your_config>.yaml --device 0 --seed 0
```

If your model has a CPU-only forward path, add a fast smoke test under `tests/`
mirroring `tests/test_hn_pcfg_smoke.py`.
