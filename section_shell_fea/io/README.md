# io — job / station JSON payloads (not a stage home)

↑ Parent: [`../README.md`](../README.md) · Computable set: [`../README.md` §2](../README.md#2-computable-set) ·
Glossary: [`../../../docs/CASCADE_GLOSSARY.md`](../../../docs/CASCADE_GLOSSARY.md)

Serialisation helpers for prepass station analysis and job outputs (PNG/JSON
payloads). **Not** the physics cascade — see `pre_processing/` / `processing/`.

## Purpose / ownership

| Owns | Does **not** own |
|------|------------------|
| `station_json`, prepass I/O, job output writers | Solve / cascade P→S→T→Q |
| Artifact paths used by benchmarks / runners | GCI maths ([`../docs/NUMERICAL_STABILITY_AND_GCI.md`](../docs/NUMERICAL_STABILITY_AND_GCI.md)) |

## See also

- [`../benchmarks/README.md`](../benchmarks/README.md)
- [`../visualisation/README.md`](../visualisation/README.md)
