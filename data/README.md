# Observation data

`observations.json` — a 12 x 10 array: 12 consecutive 30-minute time bins (rows) x 10 internal
zones (columns), recorded during a one-day university open-campus event.

## What the values are

Each value is `C[t][i]`, the estimated number of people **present in zone `i` during bin `t`** —
an occupancy, not a count of the traffic passing through the zone.

Continuous monitoring was impractical, so staff counted the ingress and egress at each zone
boundary during the **first 10 minutes of every 30-minute bin**, using manual tally counters.
Starting from an empty venue before the event opened, `C` accumulates the net crossings:

```
C[t][i] = C[t-1][i] + (ingress[t][i] - egress[t][i]),    C[0][i] = 0
```

These are the **uncorrected sums**: they are *not* rescaled to the full 30-minute bin, and are
therefore about one third of the corresponding full-bin estimate.

That constant factor is common to every zone and every time bin, and the analysis uses only the
per-bin normalised proportions

```
p[t][i] = C[t][i] / sum_j C[t][j]
```

in which it cancels exactly. No reported result depends on it.

## Privacy

No personal data were collected. Staff recorded only aggregate counts at zone boundaries, so
identification of any individual is impossible by the nature of the measurement.

See the manuscript's Methods ("Study site, observations, and time discretization") for the same
definition in context.
