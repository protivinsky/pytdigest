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
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

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

#define M_PI 3.14159265358979323846

/* TODO:
 * - python wrappers
 * - array processing functionality
 * - add variance, min, max
 * - add alternating merging procedure (should be simple, just remember what you did the last time)
 *   - actually, I need to be careful as some functions likely assume the centroids are sorted
 *   - but maybe it does not matter on the order, might be symmetric?
 *   - well not really, I would need to transform cdf and icdf - maybe simple anyway?
 * - extract scale function and allow to use a different one?
 * - add scaling of volatility?
 * - some testing and benchmarking, so I can see impact of changes
 */


typedef struct centroid {
    double mean;
    double weight;
} centroid_t;

static int centroid_compare(const void *v1, const void *v2) {
    centroid_t *c1 = (centroid_t *)(v1);
    centroid_t *c2 = (centroid_t *)(v2);
    if (c1->mean < c2->mean) {
        return -1;
    } else if (c1->mean > c2->mean) {
        return 1;
    } else {
        return 0;
    }
}

struct tdigest {
    // compression is a setting used to configure the size of centroids when merged.
    // or do I want to use delta here?
    double delta;

    // cap is the total size of nodes
    int max_centroids;
    // merged_nodes is the number of merged nodes at the front of nodes.
    int num_merged;
    // unmerged_nodes is the number of buffered nodes.
    int num_unmerged;

    double merged_weight;
    double unmerged_weight;

    centroid_t centroids[0];
};

static bool is_very_small(double val) {
    return !(val > .000000001 || val < -.000000001);
}

static int get_max_centroids(double delta) {
    // do I want to allow for parameterization here?
    return (6 * (int) (delta)) + 10;
}

static bool should_merge(tdigest_t *td) {
    return ((td->num_merged + td->num_unmerged) == td->max_centroids);
}

static int next_centroid(tdigest_t *td) {
    return td->num_merged + td->num_unmerged;
}

void merge(tdigest_t *td);


/*** CONSTRUCTORS ***/

static size_t required_buffer_size(double delta) {
    return sizeof(tdigest_t) + get_max_centroids(delta) * sizeof(centroid_t);
}

// td_init will initialize a tdigest_t inside a buffer which is buf_size bytes
// if buf_size is too small or buf is NULL, the returned pointer will be NULL
// why it is separated into two functions? I should always use td_new anyway, right:
static tdigest_t *td_init(double delta, size_t buf_size, char *buf) {
    tdigest_t *td = (tdigest_t *) (buf);
    if (!td) {
        return NULL;
    }
    bzero((void *) (td), buf_size);
    *td = (tdigest_t) {
            .delta = delta,
            .max_centroids = (buf_size - sizeof(tdigest_t)) / sizeof(centroid_t),
            .num_merged = 0,
            .merged_weight = 0,
            .num_unmerged = 0,
            .unmerged_weight = 0,
    };
    return td;
}

tdigest_t *td_new(double delta) {
    size_t memsize = required_buffer_size(delta);
    return td_init(delta, memsize, (char *) (malloc(memsize)));
}

void td_free(tdigest_t *td) {
    free((void *) (td));
}

void td_merge(tdigest_t *into, tdigest_t *from) {
    merge(into);
    merge(from);
    for (int i = 0; i < from->num_merged; i++) {
        centroid_t *c = &from->centroids[i];
        td_add(into, c->mean, c->weight);
    }
}

void td_reset(tdigest_t *td) {
    bzero((void *) (&td->centroids[0]), sizeof(centroid_t) * td->max_centroids);
    td->num_merged = 0;
    td->merged_weight = 0;
    td->num_unmerged = 0;
    td->unmerged_weight = 0;
}

// do I need this? or can I just remove it?
void td_scale_weight(tdigest_t *td, double factor) {
    merge(td);
    td->merged_weight *= factor;
    td->unmerged_weight *= factor; // this should be unnecessary, right?
    for (int i = 0; i < td->num_merged; i++) {
        td->centroids[i].weight *= factor;
    }
}

void td_shift(tdigest_t *td, double shift) {
    merge(td);
    for (int i = 0; i < td->num_merged; i++) {
        td->centroids[i].mean += shift;
    }
}

double td_total_weight(tdigest_t *td) {
    return td->merged_weight + td->unmerged_weight;
}

double td_total_sum(tdigest_t *td) {
    centroid_t *c = NULL;
    double sum = 0;
    int num_centroids = td->num_merged + td->num_unmerged;
    for (int i = 0; i < num_centroids; i++) {
        c = &td->centroids[i];
        sum += c->mean * c->weight;
    }
    return sum;
}

void td_add(tdigest_t *td, double mean, double weight) {
    if (should_merge(td)) {
        merge(td);
    }
    td->centroids[next_centroid(td)] = (centroid_t) {
        .mean = mean,
        .weight = weight,
    };
    td->num_unmerged++;
    td->unmerged_weight += weight;
}

void merge(tdigest_t *td) {
    if (td->num_unmerged == 0) {
        return;
    }
    int num_centroids = td->num_merged + td->num_unmerged;
    qsort((void *)(td->centroids), num_centroids, sizeof(centroid_t), &centroid_compare);
    double total_weight = td->merged_weight + td->unmerged_weight;
    double denominator = 2 * M_PI * total_weight * log(total_weight);
    double normalizer = td->delta / denominator;
    int cur = 0;
    double weight_so_far = 0;
    for (int i = 1; i < num_centroids; i++) {
        double proposed_weight = td->centroids[cur].weight + td->centroids[i].weight;
        double z = proposed_weight * normalizer;
        double q0 = weight_so_far / total_weight;
        double q2 = (weight_so_far + proposed_weight) / total_weight;
        bool should_add = (z <= (q0 * (1 - q0))) && (z <= (q2 * (1 - q2)));
        if (should_add) {
            td->centroids[cur].weight += td->centroids[i].weight;
            double diff = td->centroids[i].mean - td->centroids[cur].mean;
            td->centroids[cur].mean += diff * td->centroids[i].weight / td->centroids[cur].weight;
        } else {
            weight_so_far += td->centroids[cur].weight;
            cur++;
            td->centroids[cur] = td->centroids[i];
        }
        if (cur != i) {
            td->centroids[i] = (centroid_t) {
                    .mean = 0,
                    .weight = 0,
            };
        }
    }
    td->num_merged = cur + 1;
    td->merged_weight = total_weight;
    td->num_unmerged = 0;
    td->unmerged_weight = 0;
}

// this is CDF, right?
double td_quantile_of(tdigest_t *td, double val) {
    merge(td);
    if (td->num_merged == 0) {
        return NAN;
    }
    double k = 0;
    int i = 0;
    centroid_t *c = NULL;
    for (i = 0; i < td->num_merged; i++) {
        c = &td->centroids[i];
        if (c->mean >= val) {
            break;
        }
        k += c->weight;
    }
    if (val == c->mean) {
        // technically this needs to find all the nodes which contain this value and sum their weight
        double weight_at_value = c->weight;
        for (i += 1; i < td->num_merged && td->centroids[i].mean == c->mean; i++) {
            weight_at_value += td->centroids[i].weight;
        }
        return (k + (weight_at_value / 2)) / td->merged_weight;
    } else if (val > c->mean) { // past the largest
        return 1;
    } else if (i == 0) {
        return 0;
    }
    // we want to figure out where along the line from the prev node to this node, the value falls
    centroid_t *cr = c;
    centroid_t *cl = c - 1; // FIXME: is it safe? shouldn't it be sizeof or sth like that?
    k -= (cl->weight / 2);
    // we say that at zero we're at nl->mean
    // and at (nl->count/2 + nr->count/2) we're at nr
    double m = (cr->mean - cl->mean) / (cl->weight / 2 + cr->weight / 2);
    double x = (val - cl->mean) / m;
    return (k + x) / td->merged_weight;
}

// this is inverse cdf, right? or quantile, that's the same
double td_value_at(tdigest_t *td, double q) {
    merge(td);
    if (q < 0 || q > 1 || td->num_merged == 0) {
        return NAN;
    }
    // if left of the first node, use the first node
    // if right of the last node, use the last node, use it
    double goal = q * td->merged_weight;
    double k = 0;
    int i = 0;
    centroid_t *c = NULL;
    for (i = 0; i < td->num_merged; i++) {
        c = &td->centroids[i];
        if (k + c->weight > goal) {
            break;
        }
        k += c->weight;
    }
    double delta_k = goal - k - (c->weight / 2);
    if (is_very_small(delta_k)) {
        return c->mean;
    }
    bool right = delta_k > 0;
    if ((right && ((i+1) == td->num_merged)) ||
        (!right && (i == 0))) {
        return c->mean;
    }
    centroid_t *cl;
    centroid_t *cr;
    if (right) {
        cl = c;
        cr = &td->centroids[i+1];
        k += (cl->weight/2);
    } else {
        cl = &td->centroids[i-1];
        cr = c;
        k -= (cl->weight/2);
    }
    double x = goal - k;
    // we have two points (0, nl->mean), (nr->count, nr->mean)
    // and we want x
    double m = (cr->mean - cl->mean) / (cl->weight/2 + cr->weight/2);
    return m * x + cl->mean;
}

// FIXME: doesn't it assume it is sorted? but it does not have to be, right?
double td_trimmed_mean(tdigest_t *td, double lo, double hi) {
    if (should_merge(td)) {
        merge(td);
    }
    double total_weight = td->merged_weight;
    double left_tail_weight = lo * total_weight;
    double right_tail_weight = hi * total_weight;
    double count_seen = 0;
    double weighted_mean = 0;
    for (int i = 0; i < td->num_merged; i++) {
        if (i > 0) {
            count_seen += td->centroids[i-1].weight;
        }
        centroid_t *c = &td->centroids[i];
        if (c->weight < left_tail_weight) {
            continue;
        }
        if (count_seen > right_tail_weight) {
            break;
        }
        double left = count_seen;
        if (left < left_tail_weight) {
            left = left_tail_weight;
        }
        double right = count_seen + c->weight;
        if (right > right_tail_weight) {
            right = right_tail_weight;
        }
        weighted_mean += c->mean * (right - left);
    }
    double included_weight = total_weight * (hi - lo);
    return weighted_mean / included_weight;
}


void td_add_batch(tdigest_t *td, int num_values, double *means, double *weights) {
    for (int i = 0; i < num_values; i++) {
        if (should_merge(td)) {
            merge(td);
        }
        td->centroids[next_centroid(td)] = (centroid_t) {
                .mean = means[i],
                .weight = weights[i],
        };
        td->num_unmerged++;
        td->unmerged_weight += weights[i];
    }
}


void td_cdf_batch(tdigest_t *td, int count, const double *values, double *quantiles) {
    for (int i = 0; i < count; i++) {
        quantiles[i] = td_quantile_of(td, values[i]);
    }
}


void td_inverse_cdf_batch(tdigest_t *td, int count, const double *quantiles, double *values) {
    for (int i = 0; i < count; i++) {
        values[i] = td_value_at(td, quantiles[i]);
    }
}

// not necessary, it is easier to create an empty one and add existing to it
//tdigest_t *td_copy(tdigest_t *td) {
//    merge(td);
//    tdigest_t *other = td_new(td->delta);
//    other->num_merged = td->num_merged;
//    other->merged_weight = td->merged_weight;
//    for (int i = 0; i < td->num_merged; i++) {
//        other->centroids[i] = (centroid_t) {
//            .mean = td->centroids[i].mean,
//            .weight = td->centroids[i].weight,
//        };
//    }
//    return other;
//}

centroid_t *td_get_centroid(tdigest_t *td, int i) {
    if (i >= td->num_merged + td->num_unmerged) {
        return NULL;
    }
    return &td->centroids[i];
}

void td_get_centroids(tdigest_t *td, double *centroids) {
    for (int i = 0; i < td->num_merged + td->num_unmerged; i++) {
        centroids[2 * i] = td->centroids[i].mean;
        centroids[2 * i + 1] = td->centroids[i].weight;
    }
}

tdigest_t *td_of_centroids(double delta, int num_centroids, double *centroids) {
    tdigest_t *td = td_new(delta);
    double total_weight = 0;
    for (int i = 0; i < num_centroids; i++) {
        td->centroids[i] = (centroid_t) {
                    .mean = centroids[2 * i],
                    .weight = centroids[2 * i + 1],
                };
        total_weight += centroids[2 * i + 1];
    }
    td->num_unmerged = num_centroids;
    td->unmerged_weight = total_weight;
    merge(td);
    return td;
}

void td_fill_centroids(tdigest_t *td, int num_centroids, double *centroids) {
    double total_weight = 0;
    int cap = (num_centroids < td->max_centroids) ? num_centroids : td->max_centroids;
    for (int i = 0; i < cap; i++) {
        td->centroids[i] = (centroid_t) {
            .mean = centroids[2 * i],
            .weight = centroids[2 * i + 1],
        };
        total_weight += centroids[2 * i + 1];
    }
    td->num_unmerged = num_centroids;
    td->unmerged_weight = total_weight;
    merge(td);
}



//int main() {
//    printf("Hello world.\n");
//
//    tdigest_t *td = td_new(100.);
//    td_add(td, 10., 1.);
//    td_add(td, 5., 1.);
//    td_add(td, 2., 1.);
//    td_add(td, 0., 1.);
//
//    double total_weight = td_total_weight(td);
//    double total_sum = td_total_sum(td);
//
//    printf("Total weight=%f, total sum=%f", total_weight, total_sum);
//
//    td_free(td);
//
//    return 0;
//}






