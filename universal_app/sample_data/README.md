# Sample data (synthetic)

`synthetic_hourly_demo.csv` is **computer-generated**: 1,080 hourly rows of a made-up `demand_units` series (daily and
weekly cycle plus random noise) and a made-up `temperature_c` column. It is not a real measurement of anything.

`synthetic_hourly_demo_response.json` is the response the real Phase 4A engine returned for that file with
timestamp `timestamp`, target `demand_units`, horizon 24. It is stored unchanged and is replayed only in demo mode,
and only for that exact file and configuration (`synthetic_hourly_demo_meta.json` holds the fingerprint).
