#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values, or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    
    //TODO//
    
    if (rows < 1 || cols < 1) {
        PyErr_SetString(PyExc_TypeError, "Non-positive dimensions!");
        return -1;
    }
    matrix* pointer_mat = malloc(sizeof(matrix));
    if (!pointer_mat) {
        return -1;
    }
    *mat = pointer_mat;
    double* pointer_data = calloc(rows*cols, sizeof(double));
    if (!pointer_data) {
        return -1;
    }
    (pointer_mat)->data = pointer_data;
    (pointer_mat)->rows = rows;
    (pointer_mat)->cols = cols;
    (pointer_mat)->ref_cnt = 1;
    (pointer_mat)->parent = NULL;
    return 0;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`.
 * You should return -1 if either `rows` or `cols` or both are non-positive or if any
 * call to allocate memory in this function fails. Return 0 upon success.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    /* TODO: YOUR CODE HERE */
    
    if (rows < 1 || cols < 1) {
        return -1;
    }

    matrix* pointer_mat = malloc(sizeof(matrix));
    if (!pointer_mat) {
        return -1;
    }
    *mat = pointer_mat;
    (*mat)->rows = rows;
    (*mat)->cols = cols;
    (*mat)->data = from->data + offset;
    (*mat)->parent = from;
    from->ref_cnt += 1;
    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or if `mat` is the last existing slice of its parent matrix and its parent matrix has no other references
 * (including itself). You cannot assume that mat is not NULL.
 */
void deallocate_matrix(matrix *mat) {
    /* TODO: YOUR CODE HERE */
    

    if (mat) {
        matrix* par = mat->parent;
        int count = mat->ref_cnt;
        if (!par) {
            if (count == 1) {
                free(mat->data);
                free(mat);
            } else {
                mat->ref_cnt = count - 1;
            }
        } else {
            if (par->ref_cnt > 1) {
                free(mat);
                par->ref_cnt -= 1;
            } else {
                free(par->data);
                free(mat->parent);
                free(mat);
            }
        }
    }
    
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid.
 */
double get(matrix *mat, int row, int col) {
    /* TODO: YOUR CODE HERE */
    int cols = mat->cols;
    return mat->data[row*cols+col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid
 */
void set(matrix *mat, int row, int col, double val) {
    /* TODO: YOUR CODE HERE */
    int cols = mat->cols;
    mat->data[row * cols + col] = val;
}

/*
 * Sets all entries in mat to val
 */
void fill_matrix(matrix *mat, double val) {
    /* TODO: YOUR CODE HERE */
   
    int num = (mat->cols) * (mat->rows);
    double* mat_data = mat->data;
    __m256d vector = _mm256_set1_pd(val);
    int num_bottom = num/128 * 128;
    
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        _mm256_storeu_pd(mat_data + i, vector);
        _mm256_storeu_pd(mat_data + i+4, vector);
        _mm256_storeu_pd(mat_data + i+8, vector);
        _mm256_storeu_pd(mat_data + i+12, vector);
        _mm256_storeu_pd(mat_data + i+16, vector);
        _mm256_storeu_pd(mat_data + i+20, vector);
        _mm256_storeu_pd(mat_data + i+24, vector);
        _mm256_storeu_pd(mat_data + i+28, vector);
        _mm256_storeu_pd(mat_data + i+32, vector);
        _mm256_storeu_pd(mat_data + i+36, vector);
        _mm256_storeu_pd(mat_data + i+40, vector);
        _mm256_storeu_pd(mat_data + i+44, vector);
        _mm256_storeu_pd(mat_data + i+48, vector);
        _mm256_storeu_pd(mat_data + i+52, vector);
        _mm256_storeu_pd(mat_data + i+56, vector);
        _mm256_storeu_pd(mat_data + i+60, vector);
        _mm256_storeu_pd(mat_data + i+64, vector);
        _mm256_storeu_pd(mat_data + i+68, vector);
        _mm256_storeu_pd(mat_data + i+72, vector);
        _mm256_storeu_pd(mat_data + i+76, vector);
        _mm256_storeu_pd(mat_data + i+80, vector);
        _mm256_storeu_pd(mat_data + i+84, vector);
        _mm256_storeu_pd(mat_data + i+88, vector);
        _mm256_storeu_pd(mat_data + i+92, vector);
        _mm256_storeu_pd(mat_data + i+96, vector);
        _mm256_storeu_pd(mat_data + i+100, vector);
        _mm256_storeu_pd(mat_data + i+104, vector);
        _mm256_storeu_pd(mat_data + i+108, vector);
        _mm256_storeu_pd(mat_data + i+112, vector);
        _mm256_storeu_pd(mat_data + i+116, vector);
        _mm256_storeu_pd(mat_data + i+120, vector);
        _mm256_storeu_pd(mat_data + i+124, vector);
        
    }

    for (int j = num_bottom; j < num; j++) {
            mat_data[j] = val;
        }
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    
    int row_1 = result->rows;
    int col_1 = result->cols;
    int row_2 = mat1->rows;
    int col_2 = mat1->cols;
    int row_3 = mat2->rows;
    int col_3 = mat2->cols;
    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    if (row_1 != row_2 || row_2 != row_3 || col_1 != col_2 || col_2 != col_3) {
        return -1;
    }

    int num = row_1 * col_1;
    int num_bottom = num/128 * 128;
    
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        int k = i;
       __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
        __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
        __m256d vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_add_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
    }

    for (int j = num_bottom; j < num; j++) {
        result_data[j] = mat1_data[j] + mat2_data[j];
    }

    return 0;
}

/*
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    int row_1 = result->rows;
    int col_1 = result->cols;
    int row_2 = mat1->rows;
    int col_2 = mat1->cols;
    int row_3 = mat2->rows;
    int col_3 = mat2->cols;
    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    if (row_1 != row_2 || row_2 != row_3 || col_1 != col_2 || col_2 != col_3) {
        return -1;
    }
    
    int num = row_1 * col_1;
    int num_bottom = num/128 * 128;
    
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 128) {
        int k = i;
       __m256d vector1 = _mm256_loadu_pd(mat1_data + k);
        __m256d vector2 = _mm256_loadu_pd(mat2_data + k);
        __m256d vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
         vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
         vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
        k += 4;
        vector1 = _mm256_loadu_pd(mat1_data + k);
         vector2 = _mm256_loadu_pd(mat2_data + k);
        vector3 = _mm256_sub_pd(vector1, vector2);
        _mm256_storeu_pd(result_data + k, vector3);
    } 

    for (int j = num_bottom; j < num; j++) {
        result_data[j] = mat1_data[j] - mat2_data[j];
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    /* TODO: YOUR CODE HERE */
    
    int row_1 = result->rows;
  
    int col_1 = result->cols;
   
    int row_2 = mat1->rows;
  
    int col_2 = mat1->cols;
  
    int row_3 = mat2->rows;
   
    int col_3 = mat2->cols;


    if (row_1 != row_2 || col_2 != row_3 || col_3 != col_1) {
        return -1;
    }

    double* result_data = result->data;
    double* mat1_data = mat1->data;
    double* mat2_data = mat2->data;
    fill_matrix(result, 0.0);

    double* data = malloc(row_3*col_3*sizeof(double));
    matrix tempm = {col_3, row_3, data, 1, NULL};
    matrix* temp = &tempm;
    double* temp_data = temp->data;

    //transpose//

    #pragma omp parallel 
    {   
        int ave = (col_3 + row_3)/2;
        int block_size = ave/800 + 1;
        int tot_num = row_3 * col_3;
        #pragma omp for
        for (int x = 0; x < col_3/block_size + 1; x++) {
            for (int y = 0; y < row_3/block_size + 1; y++) {
                for (int i = 0; i < block_size; i++) {
                    for (int j = 0; j < block_size; j++) {
                        int a = x*row_3*block_size + y*block_size + j*row_3+i;
                        int b = x*block_size + y*col_3*block_size + i*col_3+j;
                        if (a < tot_num && b < tot_num) {
                            temp_data[a] = mat2_data[b];
                        }
                    }
                }
            }
        }
    }

    
    #pragma omp parallel 
    {
        #pragma omp for
        for (int k = 0; k < row_1; k++) {
            for (int j = 0; j < col_1; j++) {
                double* v1 = mat1_data + k * col_2;
                double* v2 = temp_data + j * row_3;
                __m256d reference = _mm256_set1_pd(0.0);
                double* entry = result_data + k * col_1 + j;
                for (int i = 0; i < col_2/32 * 32; i += 32) {
                    __m256d vector1 = _mm256_loadu_pd (v1+i);
                    __m256d vector2 = _mm256_loadu_pd (v2+i);
                    __m256d vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+4);
                    vector2 = _mm256_loadu_pd (v2+i+4);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+8);
                    vector2 = _mm256_loadu_pd (v2+i+8);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+12);
                    vector2 = _mm256_loadu_pd (v2+i+12);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+16);
                    vector2 = _mm256_loadu_pd (v2+i+16);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+20);
                    vector2 = _mm256_loadu_pd (v2+i+20);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+24);
                    vector2 = _mm256_loadu_pd (v2+i+24);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    vector1 = _mm256_loadu_pd (v1+i+28);
                    vector2 = _mm256_loadu_pd (v2+i+28);
                    vector3 = _mm256_mul_pd (vector1, vector2);
                    reference = _mm256_add_pd(reference, vector3);
                    //   vector1 = _mm256_loadu_pd (v1+i+32);
                    // vector2 = _mm256_loadu_pd (v2+i+32);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+36);
                    // vector2 = _mm256_loadu_pd (v2+i+36);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+40);
                    // vector2 = _mm256_loadu_pd (v2+i+40);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+44);
                    // vector2 = _mm256_loadu_pd (v2+i+44);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+48);
                    // vector2 = _mm256_loadu_pd (v2+i+48);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+52);
                    // vector2 = _mm256_loadu_pd (v2+i+52);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+56);
                    // vector2 = _mm256_loadu_pd (v2+i+56);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                    // vector1 = _mm256_loadu_pd (v1+i+60);
                    // vector2 = _mm256_loadu_pd (v2+i+60);
                    // vector3 = _mm256_mul_pd (vector1, vector2);
                    // reference = _mm256_add_pd(reference, vector3);
                }
                double* ptr = malloc(4*sizeof(double));
                _mm256_storeu_pd(ptr, reference);
                for (int a = 0; a < 4; a++) {
                    entry[0] += ptr[a];
                }
                free(ptr);
                for (int b = col_2/32*32; b < col_2; b++) {
                    result_data[k*col_1 + j] += v1[b] * v2[b];
                }
            }
        }
    }
    free(data);
  
    return 0;
}

/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    /* TODO: YOUR CODE HERE */
    
    if (mat->rows != mat->cols || mat->rows != result->rows || result->rows != result->cols) {
        return -1;
    }

    if (pow < 0) {
        return -1;
    }

    double* result_data = result->data;
    fill_matrix(result, 0.0);

    if (pow == 0) {
        int rows = mat->rows;
        #pragma omp parallel 
        {
        #pragma omp for
        for (int i = 0; i < rows*rows/(8*(rows+1))*8*(rows+1); i += 8*(rows+1)) {
            result_data[i] = 1.0;
            result_data[i+rows+1] = 1.0;
            result_data[i+2*rows+2] = 1.0;
            result_data[i+3*rows+3] = 1.0;
            result_data[i+4*rows+4] = 1.0;
            result_data[i+5*rows+5] = 1.0;
            result_data[i+6*rows+6] = 1.0;
            result_data[i+7*rows+7] = 1.0;
        }
        }
        for (int i = rows*rows/(8*(rows+1))*8*(rows+1); i < rows*rows; i+=(rows+1)) {
            result_data[i] = 1.0;
        }
        return 0;
    }

    int rows = mat->rows;

    add_matrix(result, result, mat);


    if (pow == 1) {
        return 0;
    }

    //helper matrix ALLOCATE!
   
    double* data = calloc(rows*rows*4, sizeof(double));
    matrix helperm = {rows, rows, data, 1, NULL};
    matrix* helper = &helperm;

    matrix tempm1 = {rows, rows, data + rows*rows, 1, NULL};
    matrix* temp1 = &tempm1;

    matrix tempm2 = {rows, rows, data + rows*rows*2, 1, NULL};
    matrix* temp2 = &tempm2;

    matrix zero = {rows, rows, data + rows*rows*3, 1, NULL};
    matrix* zeroMatrix = &zero;

    int limit = rows*rows/(8*(rows+1)) * 8*(rows+1);
    double* help = helper->data;
    // set help matrix as identity
    #pragma omp parallel 
        { 
            #pragma omp for
            for (int i = 0; i < limit; i += 8*(rows+1)) {
            help[i] = 1.0;
            help[i+rows+1] = 1.0;
            help[i+2*rows+2] = 1.0;
            help[i+3*rows+3] = 1.0;
            help[i+4*rows+4] = 1.0;
            help[i+5*rows+5] = 1.0;
            help[i+6*rows+6] = 1.0;
            help[i+7*rows+7] = 1.0;
        }
        }
    for (int i = limit; i < rows*rows; i+=(rows+1)) {
            help[i] = 1.0;
        }
    
    int i = pow;
    while (i > 1) {
        if (i%2 == 0) {
            add_matrix(temp1, zeroMatrix, result);
            // mul_matrix(temp2, temp1, temp1);
           mul_matrix(result, temp1, temp1);
            i = i/2;

        } else {
            add_matrix(temp1, zeroMatrix, helper);
            mul_matrix(helper, temp1, result);
            add_matrix(temp1, zeroMatrix, result);
            // mul_matrix(temp2, temp1, temp1);
           mul_matrix(result, temp1, temp1);
            i = (i-1)/2;

        } 
        // else if (i == 2) {
        //     add_matrix(temp1, zeroMatrix, helper);
        //    mul_matrix(helper, temp1, result);
        //     i -= 1;
        // } 
        // else {

        //     add_matrix(temp1, zeroMatrix, result);
        //     mul_matrix(temp2, temp1, temp1);  
        //    mul_matrix(result, temp1, temp2);
        //     add_matrix(temp1, zeroMatrix, helper);
        //     mul_matrix(helper, temp1, temp2);

        //     i = (i-2)/3;
        // }
    }

    add_matrix(temp1, zeroMatrix, result);
    mul_matrix(result, temp1, helper);
    free(data);
    return 0;
}

/*
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int neg_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */

    int rows = mat->rows;
    int cols = mat->cols;
    int rows_r = result->rows;
    int cols_r = result->cols;
    if (rows != rows_r || cols != cols_r) {
        return -1;
    }
    int num = rows*cols;
    double* mat_data = mat->data;
    double* result_data = result->data;
    int num_bottom = num/16 * 16;
    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 16) {
        int k = i;
        __m256d vector = _mm256_loadu_pd(mat_data+k);
        __m256d reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector = _mm256_sub_pd(reference, vector);
        _mm256_storeu_pd(result_data+k, vector);
    }
    for (int j = num_bottom; j < num; j++) {
        result_data[j] = -mat_data[j];
    }
    return 0;
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success and a nonzero value upon failure.
 */
int abs_matrix(matrix *result, matrix *mat) {
    /* TODO: YOUR CODE HERE */
 
    int rows = mat->rows;
    int cols = mat->cols;
    int rows_r = result->rows;
    int cols_r = result->cols;
    if (rows != rows_r || cols != cols_r) {
        return -1;
    }
    int num = rows*cols;
    double* mat_data = mat->data;
    double* result_data = result->data;
    int num_bottom = num/16 * 16;

    #pragma omp parallel for
    for (int i = 0; i < num_bottom; i += 16) {
        int k = i;
        __m256d vector = _mm256_loadu_pd(mat_data+k);
        __m256d reference = _mm256_set1_pd(0);
        __m256d vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
        k+=4;
         vector = _mm256_loadu_pd(mat_data+k);
         reference = _mm256_set1_pd(0);
         vector_1 = _mm256_sub_pd(reference, vector);
         vector = _mm256_max_pd (vector, vector_1);
        _mm256_storeu_pd(result_data+k, vector);
    }
    for (int j = num_bottom; j < num; j++) {
        result_data[j] = fmax(mat_data[j], -mat_data[j]);
    }
    return 0;
}

