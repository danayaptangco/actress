# Local changes to actress

Changes made to this working copy of actress (`actress_highres/`), and behaviour of the
upstream code that is worth knowing about but has **not** been changed.

Base commit: `855dc1b` ("checked inclination").

> After editing any `.py` here, re-run `setup.py`. The notebooks import the installed egg
> (`site-packages/actress-1.0-py3.9.egg`), not this source tree, so an edit has no effect
> until the egg is rebuilt.

---

## Changes made

### 2026-09-22 — limb-darkening masks are built lazily

**File:** `simulator.py`

`Simulator.__init__` used to build the three limb-darkening masks (`__photmask`,
`__spotmask`, `__facmask`) immediately, at whatever `xsize` was in force. Since `xsize`
defaults to 800 (`simulator.py:70`), the usual calling pattern

```python
sim = ac.Simulator(faculae=[], spots=[], fac_strips=[], spot_strips=[])
sim.setxsize(200)
sim.setld(ld_dict)
```

built three 800x800 masks, discarded them, rebuilt at 200x200 in `setxsize`, then
discarded and rebuilt those in `setld` — three full builds per `Simulator`, one of them
at a size never used. Each mask is an `xsize` by `xsize` meshgrid plus a limb-darkening
evaluation (`simulator.py:355-367`), so the wasted 800-sized set cost ~0.047 s. That is
negligible when a `Simulator` is built once, and dominant when one is built per
likelihood call inside a sampler.

**What changed.** The masks are now built on first use instead of eagerly:

- `__init__`, `setxsize` and `setld` set the three masks to `None` rather than building
  them.
- New private method `Simulator._ensuremasks()` builds them if they are `None`.
- `_ensuremasks()` is called at the two places the masks are read, in `makemap`'s
  consumer and in `rotate_lc`'s per-phase loop.

**Result.** Exactly one mask build per `Simulator`, at the size and coefficients actually
in force, regardless of the order the setters are called in. No caller changes are needed
anywhere.

**Verification.** Discs and Doppler shift maps compared before and after across four call
patterns (setters at xsize 200 and 100, constructor args at 200, and a bare constructor
left at the 800 default). All eight arrays bit-identical, `max abs diff = 0`.

| call pattern | before | after |
|---|---|---|
| ctor(default) + setxsize(200) + setld | 0.0628 s | 0.0104 s |
| ctor(xsize=200, ld=...) | 0.0104 s | 0.0104 s |
| ctor(default) + setxsize(100) + setld | 0.0507 s | 0.0044 s |
| bare ctor, no setxsize (800) | 0.3005 s | 0.1749 s |

(Per `rotate_lc` call including `Simulator` construction; single line, `N=1`,
`mode='faconly'`, nside 32.)

**One behavioural note.** `__photmask` and friends are `None` until the first projection.
Nothing outside `simulator.py` reads them, so this is internal only.

---

## Known upstream behaviour, NOT changed

These are properties of actress as written. They are recorded here because each one can
produce a wrong answer without raising.

### `resolution=` is accepted by the constructor and then ignored

`simulator.py:121-123` validates the `resolution` argument and then assigns
`self.__res = 2**int(6)`, hardcoded. Passing `resolution=4` gives nside 64, not 16.
`setresolution()` works correctly; the constructor argument does not.

Affected: `outputs/full_disks_and_wavelength_shifts/3_test_velocity_broadening.ipynb`
passes `resolution=RESOLUTION` and never calls `setresolution`, so it runs at nside 64.
(That notebook models a quiet star, where nside has little effect.)

### Constructor default lists are shared between instances

`simulator.py:70-71` uses mutable defaults (`faculae=[]`, `spots=[]`, `fac_strips=[]`,
`spot_strips=[]`). Python creates each of those lists once, at function-definition time,
so every `Simulator()` built without explicit lists shares them. Features added to one
instance appear in the next:

```python
a = ac.Simulator()
a.addfeature(r=5.0, lon=10.0, lat=20.0, feature='fac')
b = ac.Simulator()
b.getfill('fac')      # 0.00193, not 0.0
```

Always construct with explicit empty lists, as `run_actress_notebook.py:72` does:

```python
sim = ac.Simulator(faculae=[], spots=[], fac_strips=[], spot_strips=[])
```

Affected: `Actress_New_JT.ipynb` cell 6 builds its coverage probe as a bare
`ac.Simulator()`. Results are correct on a fresh kernel; re-running that cell in the same
kernel starts the probe already full from the previous run, so `SEED` no longer determines
the facula field. Not yet fixed.

### `addstrip` silently reflects bands that cross a pole

`simulator.py:403` passes `(lat + 90) * pi/180` to `hp.query_strip`. If a band edge exceeds
90 deg the colatitude exceeds pi, and healpy does not raise — it works in `z = cos(theta)`,
so it returns the band reflected about 90 deg. Same area, wrong latitudes:

```
query_strip(nside=64, lower=8.70, upper=131.30)  -> 14704 pixels
query_strip(nside=64, lower=8.70, upper= 48.70)  -> 14704 pixels
identical pixel sets: True
```

Because the area is preserved, `getfill()` looks correct and nothing flags the error.

**Consequence for (lat_centre, ff) parameterisations.** A band of filling factor `ff`
centred on `lat` has half-width `arcsin(ff / cos(lat))`, from `f = cos(lat) sin(h)`
(Archimedes' hat-box theorem). Two conditions are needed, not one:

- `ff <= cos(lat)` — for the arcsin to exist
- `ff <= cos(lat)**2` — for the band to stay inside +/-90 deg

The second is strictly tighter and is the one that matters. In the gap
`cos(lat)**2 < ff < cos(lat)` the model evaluated is not the one the parameters name.
Reachable for `lat` above roughly 45 deg.

Any sampler prior over `(lat_centre, ff)` should use the `cos^2` limit.
`Actress_MCMC_Timing.ipynb` uses the weaker `cos(lat)` condition in its `log_prior` and
has not been corrected. `Actress_NS_SmallSpace_Test.ipynb` uses the `cos^2` limit and
raises if a band edge leaves +/-90.
