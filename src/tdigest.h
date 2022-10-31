// Apache License, Version 2.0,
// http://www.apache.org/licenses/LICENSE-2.0
//
// Copyright (c) 2015 Ted Dunning, All rights reserved.
//      https://github.com/tdunning/t-digest
// Copyright (c) 2018 Andrew Werner, All rights reserved.
//      https://github.com/ajwerner/tdigestc
// Copyright (c) 2022 Tomas Protivinsky, All rights reserved.
//      https://github.com/protivinsky/pytdigest

#include <stdlib.h>

typedef struct tdigest tdigest_t;

// td_new allocates a new histogram.
// It is similar to init but assumes that it can use malloc.
tdigest_t *td_new(double compression);

// td_free frees the memory associated with h.
void td_free(tdigest_t *h);

// td_add adds val to h with the specified count.
void td_add(tdigest_t *h, double val, double count);

// td_merge merges the data from from into into.
void td_merge(tdigest_t *into, tdigest_t *from);

// td_reset resets a histogram.
void td_reset(tdigest_t *h);

// td_value_at queries h for the value at q.
// If q is not in [0, 1], NAN will be returned.
double td_value_at(tdigest_t *h, double q);

// td_value_at queries h for the quantile of val.
// The returned value will be in [0, 1].
double td_quantile_of(tdigest_t *h, double val);

// td_trimmed_mean returns the mean of data from the lo quantile to the
// hi quantile.
double td_trimmed_mean(tdigest_t *h, double lo, double hi);

// td_total_count returns the total count contained in h.
double td_total_weight(tdigest_t *h);

// td_total_sum returns the sum of all the data added to h.
double td_total_sum(tdigest_t *h);

// td_scale_weight multiplies all counts by factor.
void td_scale_weight(tdigest_t *h, double factor);

// td_shift shifts the whole distribution (add shift to all means)
void td_shift(tdigest_t *h, double shift);
