# Environment Specifications

The M3 Conda installation itself must not be copied to this repository. Export only the environments that were used for the paper after the final code path is confirmed.

Suggested minimum exports:

~~~bash
conda env export --from-history -n qwen > environments/qwen.yml
conda env export --no-builds -n qwen > environments/qwen-lock.yml

conda env export --from-history -n chexagent > environments/chexagent.yml
conda env export --no-builds -n chexagent > environments/chexagent-lock.yml

conda env export --from-history -n xraydar_nlp > environments/xraydar_nlp.yml
conda env export --no-builds -n xraydar_nlp > environments/xraydar_nlp-lock.yml
~~~

If Table 1 baseline scripts are released, export their verified environments separately (`vila`, `llava`, and `medgemma`) rather than combining them with the Qwen/CheXAgent environment.

Before committing, remove the final `prefix:` line from every exported YAML file and inspect any `pip:` entries for local paths, private indexes, tokens, or private package URLs.
