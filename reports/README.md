# Evaluation reports

One file per configuration, named `<split>_<prompt>_<model>_<variant>.txt`.

Reports written from the E2 model comparison onward carry a provenance header
naming the model, prompt version, preprocessing variant, temperature, and split.
Earlier reports do not.

The five headerless reports were run on the 27 example dev_eval split, before
`transient_001` to `003` were excluded (EXPERIMENTS.md E6). Their accuracy
figures are not directly comparable with the 25 example reports that followed.