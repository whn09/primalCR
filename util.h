#ifndef MATUTIL
#define MATUTIL

#include <unordered_set>
#include <numeric>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <new>  // use new[] in C++, don't use MALLOC
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <utility>
#include <map>
#include <queue>
#include <set>
#include <vector>
#include <cmath>
#include <omp.h>
#include <assert.h>

#define MALLOC(type, size) (type*)malloc(sizeof(type)*(size))

enum {ROWMAJOR, COLMAJOR};

using namespace std;
class rate_t;
class rateset_t; 
class RateComp;
class smat_t;
class testset_t;
typedef vector<double> vec_t;
typedef vector<vec_t> mat_t;
class SparseMat;


void load(const char* srcdir, smat_t &R, testset_t &T, bool with_weights = false);
void save_mat_t(mat_t A, FILE *fp, bool row_major = true);
mat_t load_mat_t(FILE *fp, bool row_major = true);
void initial(mat_t &X, long n, long k);


// implement smat class myself (liwei)
pair<double, double> compute_pairwise_error_ndcg(const mat_t& U, const mat_t& V,
        SparseMat* X, int ndcg_k);
// check comments in util.cpp in the end to know more details
mat_t read_initial(string file_name);
// return SparseMat* given smat_t
SparseMat* convert(smat_t &R);
SparseMat* convert(testset_t &T, long d1, long d2);
// return lambda * V
mat_t copy_mat_t(const mat_t& V, double lambda=1.0);
// return c * g
vec_t copy_vec_t(const vec_t& g, double c=1.0);

vec_t add_vec_vec(const vec_t& g1, const vec_t& g2, double c1, double c2);
void update_mat_add_vec(const vec_t& ui, double c, long j, mat_t& g);
void update_vec_subrange(const vec_t& ui, double c, vec_t& Ha, long Ha_start, long Ha_end);
void vectorize_mat(const mat_t& g, vec_t& res);
double vec_prod_array(const vec_t& ui, const vec_t& a, long a_start, long a_end);
void mat_substract_vec(const vec_t& delta, double s, mat_t& V);


void initial_col(mat_t &X, long k, long n);

// vector product
double dot(const vec_t &a, const vec_t &b);

// product of i-th column of matrix W and j-th column of matrix H
double dot(const mat_t &W, const int i, const mat_t &H, const int j);

// product of i-th column of matrix W and vector
double dot(const mat_t &W, const int i, const vec_t &H_j);

// l2-norm of a vector
double norm(const vec_t &a);

// Frobenious norm of a matrix
double norm(const mat_t &M);

// calculate squared loss (not useful for primalCR)
double calloss(const smat_t &R, const mat_t &W, const mat_t &H);

// caculate objective for Matrix Factorization Approach
double calobj(const smat_t &R, const mat_t &W, const mat_t &H, const double lambda, bool iscol=false);



// caculate RMSE with different input format
double calrmse(testset_t &testset, const mat_t &W, const mat_t &H, bool iscol=false);
double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &H_t);
double calrmse_r1(testset_t &testset, vec_t &Wt, vec_t &Ht, vec_t &oldWt, vec_t &oldHt);

class rate_t{
	public:
		int i, j; double v, weight;
		rate_t(int ii=0, int jj=0, double vv=0, double ww=1.0): i(ii), j(jj), v(vv), weight(ww){}
};

class entry_iterator_t{
	private:
		FILE *fp;
		char buf[1000];
	public:
		bool with_weights;
		size_t nnz;
		entry_iterator_t():nnz(0),fp(NULL), with_weights(false){}
		entry_iterator_t(size_t nnz_, const char* filename, bool with_weights_=false) {
			nnz = nnz_;
			fp = fopen(filename,"r");
			with_weights = with_weights_;
		}
		size_t size() {return nnz;}
		virtual rate_t next() {
			int i = 1, j = 1;
			double v = 0, w = 1.0;
			if (nnz > 0) {
				fgets(buf, 1000, fp);
				if (with_weights)
					sscanf(buf, "%d %d %lf %lf", &i, &j, &v, &w);
				else 
					sscanf(buf, "%d %d %lf", &i, &j, &v);
				--nnz;
			} else {
				fprintf(stderr,"Error: no more entry to iterate !!\n");
			}
            //return rate_t(i-1,j-1,v,w);
			return rate_t(i,j,v,w); // TODO for funny format (start from 0)
		}
		virtual ~entry_iterator_t(){
			if (fp) fclose(fp);
		}
};



// Comparator for sorting rates into row/column comopression storage
class SparseComp {
	public:
		const unsigned *row_idx;
		const unsigned *col_idx;
		SparseComp(const unsigned *row_idx_, const unsigned *col_idx_, bool isRCS_=true) {
			row_idx = (isRCS_)? row_idx_: col_idx_;
			col_idx = (isRCS_)? col_idx_: row_idx_;
		}
		bool operator()(size_t x, size_t y) const {
			return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]<= col_idx[y]));
		}
};



// Access column fomat only when you use it..
class smat_t{
	public:
		long rows, cols;
		long nnz, max_row_nnz, max_col_nnz;
		double *val, *val_t;
		double *weight, *weight_t;
		long *col_ptr, *row_ptr;
		long *col_nnz, *row_nnz;
		unsigned *row_idx, *col_idx;    // condensed
		//unsigned long *row_idx, *col_idx; // for matlab
		bool mem_alloc_by_me, with_weights;
		smat_t():mem_alloc_by_me(false), with_weights(false){ }
		smat_t(const smat_t& m){ *this = m; mem_alloc_by_me = false;}

		// For matlab (Almost deprecated)
		smat_t(long m, long n, unsigned *ir, long *jc, double *v, unsigned *ir_t, long *jc_t, double *v_t):
		//smat_t(long m, long n, unsigned long *ir, long *jc, double *v, unsigned long *ir_t, long *jc_t, double *v_t):
			rows(m), cols(n), mem_alloc_by_me(false), 
			row_idx(ir), col_ptr(jc), val(v), col_idx(ir_t), row_ptr(jc_t), val_t(v_t) {
			if(col_ptr[n] != row_ptr[m]) 
				fprintf(stderr,"Error occurs! two nnz do not match (%ld, %ld)\n", col_ptr[n], row_ptr[n]);
			nnz = col_ptr[n];
			max_row_nnz = max_col_nnz = 0;
			for(long r=1; r<=rows; ++r)
				max_row_nnz = max(max_row_nnz, row_ptr[r]);
			for(long c=1; c<=cols; ++c) 
				max_col_nnz = max(max_col_nnz, col_ptr[c]);
		}

		void from_mpi(){
			mem_alloc_by_me=true;
			max_col_nnz = 0;
			for(long c=1; c<=cols; ++c) 
				max_col_nnz = max(max_col_nnz, col_ptr[c]-col_ptr[c-1]);
		}
		void print_mat(int host){
			for(int c = 0; c < cols; ++c) if(col_ptr[c+1]>col_ptr[c]){
				printf("%d: %ld at host %d\n", c, col_ptr[c+1]-col_ptr[c],host);
			}
		}
		void load(long _rows, long _cols, long _nnz, const char* filename, bool with_weights = false){
			entry_iterator_t entry_it(_nnz, filename, with_weights);
			load_from_iterator(_rows, _cols, _nnz, &entry_it);
		}
		void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it) {
			rows =_rows,cols=_cols,nnz=_nnz;
            cout << "rows = " << rows << ", cols = " << cols << ", nnz = " << nnz << endl;
			mem_alloc_by_me = true;
			with_weights = entry_it->with_weights;
			val = MALLOC(double, nnz); val_t = MALLOC(double, nnz);
			if(with_weights) { weight = MALLOC(double, nnz); weight_t = MALLOC(double, nnz); }
			row_idx = MALLOC(unsigned, nnz); col_idx = MALLOC(unsigned, nnz);  // switch to this for memory
			row_ptr = MALLOC(long, rows+1); col_ptr = MALLOC(long, cols+1);
			memset(row_ptr,0,sizeof(long)*(rows+1));
			memset(col_ptr,0,sizeof(long)*(cols+1));

			/*
			 * Assume ratings are stored in the row-majored ordering
			for(size_t idx = 0; idx < _nnz; idx++){
				rate_t rate = entry_it->next();
				row_ptr[rate.i+1]++;
				col_ptr[rate.j+1]++;
				col_idx[idx] = rate.j;
				val_t[idx] = rate.v;
			}*/

			// a trick here to utilize the space the have been allocated 
			vector<size_t> perm(_nnz);
			unsigned *tmp_row_idx = col_idx;
			unsigned *tmp_col_idx = row_idx;
			double *tmp_val = val;
			double *tmp_weight = weight;
			for(size_t idx = 0; idx < _nnz; idx++){
				rate_t rate = entry_it->next();
                //cout << "rate.i = " << rate.i << ", rate.j = " << rate.j << ", rate.v = " << rate.v << endl;
				row_ptr[rate.i+1]++;
				col_ptr[rate.j+1]++;
				tmp_row_idx[idx] = rate.i; 
				tmp_col_idx[idx] = rate.j;
				tmp_val[idx] = rate.v;
				if(with_weights) 
					tmp_weight[idx] = rate.weight;
				perm[idx] = idx;
			}
			// sort entries into row-majored ordering
			sort(perm.begin(), perm.end(), SparseComp(tmp_row_idx, tmp_col_idx, true));
			// Generate CRS format
			for(size_t idx = 0; idx < _nnz; idx++) {
				val_t[idx] = tmp_val[perm[idx]];
				col_idx[idx] = tmp_col_idx[perm[idx]];
				if(with_weights)
					weight_t[idx] = tmp_weight[idx];
			}

			// Calculate nnz for each row and col
			max_row_nnz = max_col_nnz = 0;
			for(long r=1; r<=rows; ++r) {
				max_row_nnz = max(max_row_nnz, row_ptr[r]);
				row_ptr[r] += row_ptr[r-1];
			}
			for(long c=1; c<=cols; ++c) {
				max_col_nnz = max(max_col_nnz, col_ptr[c]);
				col_ptr[c] += col_ptr[c-1];
			}
			// Transpose CRS into CCS matrix
			for(long r=0; r<rows; ++r){
				for(long i = row_ptr[r]; i < row_ptr[r+1]; ++i){
					long c = col_idx[i];
					row_idx[col_ptr[c]] = r; 
					val[col_ptr[c]] = val_t[i];
					if(with_weights) weight[col_ptr[c]] = weight_t[i];	
					col_ptr[c]++;
				}
			}
			for(long c=cols; c>0; --c) col_ptr[c] = col_ptr[c-1];
			col_ptr[0] = 0;
            cout << "load done." << endl;
		}
		long nnz_of_row(int i) const {return (row_ptr[i+1]-row_ptr[i]);}
		long nnz_of_col(int i) const {return (col_ptr[i+1]-col_ptr[i]);}
		double get_global_mean(){
			double sum=0;
			for(long i=0;i<nnz;++i) sum+=val[i];
			return sum/nnz;
		}
		void remove_bias(double bias=0){
			if(bias) {
				for(long i=0;i<nnz;++i) val[i]-=bias;
				for(long i=0;i<nnz;++i) val_t[i]-=bias;
			}
		}
		void free(void *ptr) {if(ptr) ::free(ptr);}
		~smat_t(){
			if(mem_alloc_by_me) {
				//puts("Warnning: Somebody just free me.");
				free(val); free(val_t);
				free(row_ptr);free(row_idx); 
				free(col_ptr);free(col_idx);
				if(with_weights) { free(weight); free(weight_t);}
			}
		}
		void clear_space() {
			free(val); free(val_t);
			free(row_ptr);free(row_idx); 
			free(col_ptr);free(col_idx);
			if(with_weights) { free(weight); free(weight_t);}
			mem_alloc_by_me = false;
			with_weights = false;

		}
		smat_t transpose(){
			smat_t mt;
			mt.cols = rows; mt.rows = cols; mt.nnz = nnz;
			mt.val = val_t; mt.val_t = val;
			mt.with_weights = with_weights;
			mt.weight = weight_t; mt.weight_t = weight;
			mt.col_ptr = row_ptr; mt.row_ptr = col_ptr;
			mt.col_idx = row_idx; mt.row_idx = col_idx;
			mt.max_col_nnz=max_row_nnz; mt.max_row_nnz=max_col_nnz;
			return mt;
		}
};


// row-major iterator
class smat_iterator_t: public entry_iterator_t{
	private:
		unsigned *col_idx;
		long *row_ptr;
		double *val_t;
		double *weight_t;
		size_t	rows, cols, cur_idx, cur_row;
		bool with_weights;
	public:
		smat_iterator_t(const smat_t& M, int major = ROWMAJOR) {
			nnz = M.nnz;
			col_idx = (major == ROWMAJOR)? M.col_idx: M.row_idx;
			row_ptr = (major == ROWMAJOR)? M.row_ptr: M.col_ptr;
			val_t = (major == ROWMAJOR)? M.val_t: M.val;
			weight_t = (major == ROWMAJOR)? M.weight_t: M.weight; 
			with_weights = M.with_weights;
			rows = (major==ROWMAJOR)? M.rows: M.cols;
			cols = (major==ROWMAJOR)? M.cols: M.rows;
			cur_idx = cur_row = 0;
		}
		~smat_iterator_t() {}
		rate_t next() {
			int i = 1, j = 1;
			double v = 0;
			while (cur_idx >= row_ptr[cur_row+1]) ++cur_row;
			if (nnz > 0) --nnz;
			else fprintf(stderr,"Error: no more entry to iterate !!\n");
			rate_t ret(cur_row, col_idx[cur_idx], val_t[cur_idx], with_weights? weight_t[cur_idx]: 1.0);
			cur_idx++;
			return ret;
		}
};


// Test set format
class testset_t{
	public:
	long rows, cols, nnz;
	vector<rate_t> T;
	testset_t(): rows(0), cols(0), nnz(0){}
	inline rate_t& operator[](const unsigned &idx) {return T[idx];}
	void load(long _rows, long _cols, long _nnz, const char *filename) {
		int r, c; 
		double v;
		rows = _rows; cols = _cols; nnz = _nnz;
		T = vector<rate_t>(nnz);
		FILE *fp = fopen(filename, "r");
		for(long idx = 0; idx < nnz; ++idx){
			fscanf(fp, "%d %d %lg", &r, &c, &v); 
			//T[idx] = rate_t(r-1,c-1,v);
            T[idx] = rate_t(r,c,v); // TODO for funny format (start from 0)
		}
		fclose(fp);
	}
	void load_from_iterator(long _rows, long _cols, long _nnz, entry_iterator_t* entry_it){ 
		rows =_rows,cols=_cols,nnz=_nnz;
		T = vector<rate_t>(nnz);
		for(size_t idx=0; idx < nnz; ++idx) 
			T[idx] = entry_it->next();
	}
	double get_global_mean(){
		double sum=0;
		for(long i=0; i<nnz; ++i) sum+=T[i].v;
		return sum/nnz;
	}
	void remove_bias(double bias=0){
		if(bias) for(long i=0; i<nnz; ++i) T[i].v-=bias;
	}
};


// smat is d2 by d1 sparse matrix, where d1 is number of users and d2 is number of movies
class SparseMat{
    public:
        long d1, d2;
        long nnz;
        double *vals;
        long *index;
        long *cols;
        long *rows;
        SparseMat(long d1, long d2, long nnz){
            this->d1 = d1;
			this->d2 = d2;
			this->nnz = nnz;
			vals = new double[nnz];
            index = new long[d1 + 1];
            cols = new long[nnz];
            rows = new long[nnz];
        }
        ~SparseMat(){
            delete[] vals;
            delete[] index;
            delete[] cols;
            delete[] rows;
        }
};
#endif
