#ifndef SORTING_HELPERS_H
#define SORTING_HELPERS_H

int asc_compare(const void *a, const void *b);
int desc_compare(const void *a, const void *b);
int array_compare(int *arr1, int *arr2, int n);

int asc_compare(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int desc_compare(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}

int array_compare(int *arr1, int *arr2, int n) {

    int i;

    for (i = 0; i < n; i++) {
        if (arr1[i] != arr2[i]) {
            return 0;
        }
    }

    return 1;
}

#endif // SORTING_HELPERS_H