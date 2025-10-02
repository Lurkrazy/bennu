#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
#if ((__CUDACC_VER_MAJOR__ < 12) || ((__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ < 8)))
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
#endif
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

};
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}
#include <mma.h>

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ A, half* __restrict__ W, half* __restrict__ conv2d_nchw);
extern "C" __global__ void __launch_bounds__(64) main_kernel(half* __restrict__ A, half* __restrict__ W, half* __restrict__ conv2d_nchw) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_nchw_reindex_shared_dyn_wmma_accumulator[8];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> W_reindex_pad_shared_dyn_wmma_matrix_b[4];
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], 0.000000e+00f);
  half condval;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 675)];
  } else {
    condval = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval;
  half condval_1;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_1 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 674)];
  } else {
    condval_1 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_1;
  half condval_2;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_2 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 673)];
  } else {
    condval_2 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_2;
  half condval_3;
  if ((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_3 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 672)];
  } else {
    condval_3 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_3;
  half condval_4;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_4 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 671)];
  } else {
    condval_4 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_4;
  half condval_5;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_5 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 670)];
  } else {
    condval_5 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_5;
  half condval_6;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 6) / 7))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_6 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 6) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_6 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_6;
  half condval_7;
  if (((2 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_7 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 451)];
  } else {
    condval_7 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_7;
  half condval_8;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_8 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 675)];
  } else {
    condval_8 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_8;
  half condval_9;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_9 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 674)];
  } else {
    condval_9 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_9;
  half condval_10;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_10 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 673)];
  } else {
    condval_10 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_10;
  half condval_11;
  if ((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_11 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 672)];
  } else {
    condval_11 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_11;
  half condval_12;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_12 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 671)];
  } else {
    condval_12 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_12;
  half condval_13;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_13 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 670)];
  } else {
    condval_13 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_13;
  half condval_14;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 6) / 7))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_14 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 6) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_14 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_14;
  half condval_15;
  if (((2 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_15 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 451)];
  } else {
    condval_15 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_15;
  int4 v_ = make_int4((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)))+(147*0), (((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)))+(147*1), (((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)))+(147*2), (((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v_.x],W[v_.y],W[v_.z],W[v_.w]);
  int4 v__1 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 4))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 4))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 4))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 4))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__1.x],W[v__1.y],W[v__1.z],W[v__1.w]);
  int4 v__2 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 8))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 8))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 8))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 8))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__2.x],W[v__2.y],W[v__2.z],W[v__2.w]);
  int4 v__3 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 12))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 12))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 12))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 12))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__3.x],W[v__3.y],W[v__3.z],W[v__3.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_16;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_16 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 225)];
  } else {
    condval_16 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_16;
  half condval_17;
  if ((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_17 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 224)];
  } else {
    condval_17 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_17;
  half condval_18;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_18 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 223)];
  } else {
    condval_18 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_18;
  half condval_19;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_19 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 222)];
  } else {
    condval_19 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_19;
  half condval_20;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_20 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_20 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_20;
  half condval_21;
  if ((3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_21 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 3)];
  } else {
    condval_21 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_21;
  half condval_22;
  if ((2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_22 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 2)];
  } else {
    condval_22 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_22;
  half condval_23;
  if ((1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_23 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) - 1)];
  } else {
    condval_23 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_23;
  half condval_24;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_24 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 225)];
  } else {
    condval_24 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_24;
  half condval_25;
  if ((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_25 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 224)];
  } else {
    condval_25 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_25;
  half condval_26;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_26 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 223)];
  } else {
    condval_26 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_26;
  half condval_27;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_27 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 222)];
  } else {
    condval_27 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_27;
  half condval_28;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_28 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_28 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_28;
  half condval_29;
  if ((3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_29 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 3)];
  } else {
    condval_29 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_29;
  half condval_30;
  if ((2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_30 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 2)];
  } else {
    condval_30 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_30;
  half condval_31;
  if ((1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_31 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) - 1)];
  } else {
    condval_31 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_31;
  int4 v__4 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 16))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 16))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 16))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 16))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__4.x],W[v__4.y],W[v__4.z],W[v__4.w]);
  int4 v__5 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 20))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 20))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 20))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 20))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__5.x],W[v__5.y],W[v__5.z],W[v__5.w]);
  int4 v__6 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 24))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 24))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 24))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 24))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__6.x],W[v__6.y],W[v__6.z],W[v__6.w]);
  int4 v__7 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 28))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 28))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 28))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 28))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__7.x],W[v__7.y],W[v__7.z],W[v__7.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_32;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_32 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 225)];
  } else {
    condval_32 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_32;
  half condval_33;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_33 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 226)];
  } else {
    condval_33 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_33;
  half condval_34;
  if (((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_34 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_34 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_34;
  half condval_35;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_35 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 445)];
  } else {
    condval_35 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_35;
  half condval_36;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_36 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 446)];
  } else {
    condval_36 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_36;
  half condval_37;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_37 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 447)];
  } else {
    condval_37 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_37;
  half condval_38;
  if ((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111)) {
    condval_38 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 448)];
  } else {
    condval_38 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_38;
  half condval_39;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_39 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 449)];
  } else {
    condval_39 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_39;
  half condval_40;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_40 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 225)];
  } else {
    condval_40 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_40;
  half condval_41;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_41 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 226)];
  } else {
    condval_41 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_41;
  half condval_42;
  if (((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_42 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_42 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_42;
  half condval_43;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_43 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 445)];
  } else {
    condval_43 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_43;
  half condval_44;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_44 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 446)];
  } else {
    condval_44 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_44;
  half condval_45;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_45 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 447)];
  } else {
    condval_45 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_45;
  half condval_46;
  if ((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111)) {
    condval_46 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 448)];
  } else {
    condval_46 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_46;
  half condval_47;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_47 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 449)];
  } else {
    condval_47 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_47;
  int4 v__8 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 32))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 32))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 32))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 32))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__8.x],W[v__8.y],W[v__8.z],W[v__8.w]);
  int4 v__9 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 36))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 36))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 36))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 36))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__9.x],W[v__9.y],W[v__9.z],W[v__9.w]);
  int4 v__10 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 40))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 40))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 40))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 40))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__10.x],W[v__10.y],W[v__10.z],W[v__10.w]);
  int4 v__11 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 44))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 44))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 44))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 44))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__11.x],W[v__11.y],W[v__11.z],W[v__11.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_48;
  if (((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_48 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 48) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_48 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_48;
  half condval_49;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_49 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49501)];
  } else {
    condval_49 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_49;
  half condval_50;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_50 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49502)];
  } else {
    condval_50 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_50;
  half condval_51;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_51 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49503)];
  } else {
    condval_51 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_51;
  half condval_52;
  if ((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_52 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49504)];
  } else {
    condval_52 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_52;
  half condval_53;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_53 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49505)];
  } else {
    condval_53 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_53;
  half condval_54;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_54 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49506)];
  } else {
    condval_54 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_54;
  half condval_55;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 6) / 7))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_55 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 6) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_55 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_55;
  half condval_56;
  if (((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_56 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 48) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_56 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_56;
  half condval_57;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_57 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49501)];
  } else {
    condval_57 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_57;
  half condval_58;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_58 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49502)];
  } else {
    condval_58 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_58;
  half condval_59;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_59 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49503)];
  } else {
    condval_59 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_59;
  half condval_60;
  if ((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_60 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49504)];
  } else {
    condval_60 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_60;
  half condval_61;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_61 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49505)];
  } else {
    condval_61 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_61;
  half condval_62;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_62 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49506)];
  } else {
    condval_62 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_62;
  half condval_63;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 6) / 7))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_63 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 6) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_63 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_63;
  int4 v__12 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 48))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 48))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 48))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 48))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__12.x],W[v__12.y],W[v__12.z],W[v__12.w]);
  int4 v__13 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 52))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 52))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 52))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 52))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__13.x],W[v__13.y],W[v__13.z],W[v__13.w]);
  int4 v__14 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 56))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 56))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 56))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 56))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__14.x],W[v__14.y],W[v__14.z],W[v__14.w]);
  int4 v__15 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 60))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 60))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 60))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 60))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__15.x],W[v__15.y],W[v__15.z],W[v__15.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_64;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_64 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49950)];
  } else {
    condval_64 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_64;
  half condval_65;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_65 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49951)];
  } else {
    condval_65 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_65;
  half condval_66;
  if ((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_66 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49952)];
  } else {
    condval_66 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_66;
  half condval_67;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_67 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49953)];
  } else {
    condval_67 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_67;
  half condval_68;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_68 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 49954)];
  } else {
    condval_68 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_68;
  half condval_69;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_69 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_69 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_69;
  half condval_70;
  if ((3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_70 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50173)];
  } else {
    condval_70 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_70;
  half condval_71;
  if ((2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_71 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50174)];
  } else {
    condval_71 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_71;
  half condval_72;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_72 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49950)];
  } else {
    condval_72 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_72;
  half condval_73;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_73 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49951)];
  } else {
    condval_73 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_73;
  half condval_74;
  if ((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_74 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49952)];
  } else {
    condval_74 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_74;
  half condval_75;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_75 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49953)];
  } else {
    condval_75 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_75;
  half condval_76;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_76 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 49954)];
  } else {
    condval_76 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_76;
  half condval_77;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_77 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_77 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_77;
  half condval_78;
  if ((3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_78 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50173)];
  } else {
    condval_78 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_78;
  half condval_79;
  if ((2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_79 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50174)];
  } else {
    condval_79 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_79;
  int4 v__16 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 64))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 64))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 64))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 64))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__16.x],W[v__16.y],W[v__16.z],W[v__16.w]);
  int4 v__17 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 68))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 68))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 68))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 68))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__17.x],W[v__17.y],W[v__17.z],W[v__17.w]);
  int4 v__18 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 72))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 72))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 72))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 72))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__18.x],W[v__18.y],W[v__18.z],W[v__18.w]);
  int4 v__19 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 76))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 76))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 76))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 76))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__19.x],W[v__19.y],W[v__19.z],W[v__19.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_80;
  if (((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223)) {
    condval_80 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50400)];
  } else {
    condval_80 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_80;
  half condval_81;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_81 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50401)];
  } else {
    condval_81 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_81;
  half condval_82;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_82 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50402)];
  } else {
    condval_82 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_82;
  half condval_83;
  if (((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_83 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_83 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_83;
  half condval_84;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_84 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50621)];
  } else {
    condval_84 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_84;
  half condval_85;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_85 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50622)];
  } else {
    condval_85 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_85;
  half condval_86;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_86 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50623)];
  } else {
    condval_86 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_86;
  half condval_87;
  if ((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111)) {
    condval_87 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 50624)];
  } else {
    condval_87 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_87;
  half condval_88;
  if (((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223)) {
    condval_88 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50400)];
  } else {
    condval_88 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_88;
  half condval_89;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_89 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50401)];
  } else {
    condval_89 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_89;
  half condval_90;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_90 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50402)];
  } else {
    condval_90 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_90;
  half condval_91;
  if (((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_91 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 49501)];
  } else {
    condval_91 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_91;
  half condval_92;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_92 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50621)];
  } else {
    condval_92 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_92;
  half condval_93;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_93 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50622)];
  } else {
    condval_93 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_93;
  half condval_94;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_94 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50623)];
  } else {
    condval_94 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_94;
  half condval_95;
  if ((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111)) {
    condval_95 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 50624)];
  } else {
    condval_95 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_95;
  int4 v__20 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 80))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 80))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 80))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 80))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__20.x],W[v__20.y],W[v__20.z],W[v__20.w]);
  int4 v__21 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 84))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 84))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 84))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 84))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__21.x],W[v__21.y],W[v__21.z],W[v__21.w]);
  int4 v__22 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 88))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 88))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 88))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 88))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__22.x],W[v__22.y],W[v__22.z],W[v__22.w]);
  int4 v__23 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 92))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 92))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 92))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 92))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__23.x],W[v__23.y],W[v__23.z],W[v__23.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_96;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)) < 227)) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_96 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 96) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((int)threadIdx.x) & 1)) - 670)];
  } else {
    condval_96 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_96;
  half condval_97;
  if (((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_97 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 97) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_97 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_97;
  half condval_98;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_98 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99677)];
  } else {
    condval_98 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_98;
  half condval_99;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_99 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99678)];
  } else {
    condval_99 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_99;
  half condval_100;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_100 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99679)];
  } else {
    condval_100 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_100;
  half condval_101;
  if ((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_101 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99680)];
  } else {
    condval_101 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_101;
  half condval_102;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_102 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99681)];
  } else {
    condval_102 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_102;
  half condval_103;
  if (((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_103 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 99682)];
  } else {
    condval_103 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_103;
  half condval_104;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)) < 227)) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_104 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 96) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((int)threadIdx.x) & 1)) - 670)];
  } else {
    condval_104 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_104;
  half condval_105;
  if (((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_105 = A[(((((((((((((int)threadIdx.x) & 1) * 8) + 97) / 49) * 50176) + (((int)blockIdx.y) * 1792)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) - 675)];
  } else {
    condval_105 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_105;
  half condval_106;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_106 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99677)];
  } else {
    condval_106 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_106;
  half condval_107;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_107 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99678)];
  } else {
    condval_107 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_107;
  half condval_108;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_108 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99679)];
  } else {
    condval_108 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_108;
  half condval_109;
  if ((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_109 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99680)];
  } else {
    condval_109 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_109;
  half condval_110;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_110 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99681)];
  } else {
    condval_110 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_110;
  half condval_111;
  if (((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_111 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 99682)];
  } else {
    condval_111 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_111;
  int4 v__24 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 96))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 96))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 96))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 96))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__24.x],W[v__24.y],W[v__24.z],W[v__24.w]);
  int4 v__25 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 100))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 100))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 100))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 100))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__25.x],W[v__25.y],W[v__25.z],W[v__25.w]);
  int4 v__26 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 104))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 104))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 104))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 104))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__26.x],W[v__26.y],W[v__26.z],W[v__26.w]);
  int4 v__27 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 108))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 108))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 108))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 108))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__27.x],W[v__27.y],W[v__27.z],W[v__27.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_112;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_112 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100125)];
  } else {
    condval_112 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_112;
  half condval_113;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_113 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100126)];
  } else {
    condval_113 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_113;
  half condval_114;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_114 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100127)];
  } else {
    condval_114 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_114;
  half condval_115;
  if ((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_115 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100128)];
  } else {
    condval_115 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_115;
  half condval_116;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_116 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100129)];
  } else {
    condval_116 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_116;
  half condval_117;
  if (((1 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_117 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100130)];
  } else {
    condval_117 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_117;
  half condval_118;
  if ((((3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_118 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 99677)];
  } else {
    condval_118 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_118;
  half condval_119;
  if ((3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_119 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100349)];
  } else {
    condval_119 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_119;
  half condval_120;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_120 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100125)];
  } else {
    condval_120 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_120;
  half condval_121;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_121 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100126)];
  } else {
    condval_121 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_121;
  half condval_122;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_122 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100127)];
  } else {
    condval_122 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_122;
  half condval_123;
  if ((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)))) {
    condval_123 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100128)];
  } else {
    condval_123 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_123;
  half condval_124;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_124 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100129)];
  } else {
    condval_124 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_124;
  half condval_125;
  if (((1 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1))) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_125 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100130)];
  } else {
    condval_125 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_125;
  half condval_126;
  if ((((3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 20) / 7))) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_126 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 20) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 99677)];
  } else {
    condval_126 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_126;
  half condval_127;
  if ((3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)))) {
    condval_127 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100349)];
  } else {
    condval_127 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_127;
  int4 v__28 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 112))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 112))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 112))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 112))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__28.x],W[v__28.y],W[v__28.z],W[v__28.w]);
  int4 v__29 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 116))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 116))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 116))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 116))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__29.x],W[v__29.y],W[v__29.z],W[v__29.w]);
  int4 v__30 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 120))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 120))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 120))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 120))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__30.x],W[v__30.y],W[v__30.z],W[v__30.w]);
  int4 v__31 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 124))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 124))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 124))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 124))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__31.x],W[v__31.y],W[v__31.z],W[v__31.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_128;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_128 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100575)];
  } else {
    condval_128 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_128;
  half condval_129;
  if (((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223)) {
    condval_129 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100576)];
  } else {
    condval_129 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_129;
  half condval_130;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_130 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100577)];
  } else {
    condval_130 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_130;
  half condval_131;
  if ((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_131 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100578)];
  } else {
    condval_131 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = condval_131;
  half condval_132;
  if (((((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_132 = A[((((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 99677)];
  } else {
    condval_132 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = condval_132;
  half condval_133;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_133 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100797)];
  } else {
    condval_133 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = condval_133;
  half condval_134;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (2 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_134 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100798)];
  } else {
    condval_134 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = condval_134;
  half condval_135;
  if (((((((int)blockIdx.y) * 4) + ((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112)) < 111) && (1 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_135 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 100799)];
  } else {
    condval_135 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = condval_135;
  half condval_136;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_136 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100575)];
  } else {
    condval_136 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_136;
  half condval_137;
  if (((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223)) {
    condval_137 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100576)];
  } else {
    condval_137 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_137;
  half condval_138;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_138 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100577)];
  } else {
    condval_138 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_138;
  half condval_139;
  if ((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((int)threadIdx.x) & 1)) < 223) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_139 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100578)];
  } else {
    condval_139 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = condval_139;
  half condval_140;
  if (((((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + ((((((int)threadIdx.x) & 1) * 8) + 34) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_140 = A[((((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + (((((((int)threadIdx.x) & 1) * 8) + 34) / 7) * 224)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + (((((int)threadIdx.x) & 1) + 6) % 7)) + 99677)];
  } else {
    condval_140 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = condval_140;
  half condval_141;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_141 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100797)];
  } else {
    condval_141 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = condval_141;
  half condval_142;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (2 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_142 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100798)];
  } else {
    condval_142 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = condval_142;
  half condval_143;
  if (((((((int)blockIdx.y) * 4) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112)) < 111) && (1 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1))))) {
    condval_143 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 100799)];
  } else {
    condval_143 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = condval_143;
  int4 v__32 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 128))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 128))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 128))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 128))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = make_half4(W[v__32.x],W[v__32.y],W[v__32.z],W[v__32.w]);
  int4 v__33 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 132))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 132))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 132))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 132))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(W[v__33.x],W[v__33.y],W[v__33.z],W[v__33.w]);
  int4 v__34 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 136))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 136))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 136))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 136))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(W[v__34.x],W[v__34.y],W[v__34.z],W[v__34.w]);
  int4 v__35 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 140))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 140))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 140))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 140))+(147*3));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(W[v__35.x],W[v__35.y],W[v__35.z],W[v__35.w]);
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  half condval_144;
  if ((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 46) % 49) / 7)) < 227)) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_144 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 101025)];
  } else {
    condval_144 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1152)] = condval_144;
  half condval_145;
  if ((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)) < 227)) && (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) < 111))) {
    condval_145 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 101026)];
  } else {
    condval_145 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1153)] = condval_145;
  half condval_146;
  if (((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_146 = A[(((((((int)blockIdx.y) * 1792) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + (((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) % 112) * 2)) + 101027)];
  } else {
    condval_146 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1154)] = condval_146;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1155)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1156)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1157)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1158)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 1159)] = __float2half_rn(0.000000e+00f);
  half condval_147;
  if ((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 46) % 49) / 7)) < 227)) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((int)threadIdx.x) & 1)) < 223))) {
    condval_147 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 101025)];
  } else {
    condval_147 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2432)] = condval_147;
  half condval_148;
  if ((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 47) % 49) / 7)) < 227)) && ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) < 111))) {
    condval_148 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 101026)];
  } else {
    condval_148 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2433)] = condval_148;
  half condval_149;
  if (((((((((int)threadIdx.x) & 1) < 1) && (3 <= (((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)))) && ((((((int)blockIdx.y) * 8) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 2)) + (((((((int)threadIdx.x) & 1) * 8) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)))) && ((((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2) + (((((int)threadIdx.x) & 1) + 6) % 7)) < 227))) {
    condval_149 = A[(((((((int)blockIdx.y) * 1792) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) / 112) * 448)) + ((((int)threadIdx.x) & 1) * 225)) + ((((((((int)blockIdx.x) * 64) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) >> 1)) + 32) % 112) * 2)) + 101027)];
  } else {
    condval_149 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2434)] = condval_149;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2435)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2436)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2437)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2438)] = __float2half_rn(0.000000e+00f);
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.y) * 640) + ((((int)threadIdx.x) >> 1) * 40)) + ((((int)threadIdx.x) & 1) * 8)) + 2439)] = __float2half_rn(0.000000e+00f);
  half4 condval_150;
  if ((((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 4)) < 3)) {
    int4 v__36 = make_int4(((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 144))+(147*0), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 144))+(147*1), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 144))+(147*2), ((((((((int)threadIdx.x) & 15) * 588) + (((int)threadIdx.y) * 2)) + (((int)threadIdx.x) >> 4)) + 144))+(147*3));
    condval_150 = make_half4(W[v__36.x],W[v__36.y],W[v__36.z],W[v__36.w]);
  } else {
    condval_150 = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  }
  *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4))) = condval_150;
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 288)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 576)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  *(half4*)(((half*)buf_dyn_shmem) + ((((((int)threadIdx.y) * 144) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)) + 864)) = make_half4(__float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f), __float2half_rn(0.000000e+00f));
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1152)])), 40);
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], (&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1280) + 1792)])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[0])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[16])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[2], (&(((half*)buf_dyn_shmem)[32])), 72);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[3], (&(((half*)buf_dyn_shmem)[48])), 72);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[2], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1], W_reindex_pad_shared_dyn_wmma_matrix_b[3], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1152)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1408)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1664)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1920)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[3], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
    int4 v__37 = make_int4((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)))+(12544*0), (((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)))+(12544*1), (((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)))+(12544*2), (((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)))+(12544*3));
    half4 v__38 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1152));
    conv2d_nchw[v__37.x] = v__38.x;
    conv2d_nchw[v__37.y] = v__38.y;
    conv2d_nchw[v__37.z] = v__38.z;
    conv2d_nchw[v__37.w] = v__38.w;
    int4 v__39 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200704))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200704))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200704))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200704))+(12544*3));
    half4 v__40 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1408));
    conv2d_nchw[v__39.x] = v__40.x;
    conv2d_nchw[v__39.y] = v__40.y;
    conv2d_nchw[v__39.z] = v__40.z;
    conv2d_nchw[v__39.w] = v__40.w;
    int4 v__41 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401408))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401408))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401408))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401408))+(12544*3));
    half4 v__42 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1664));
    conv2d_nchw[v__41.x] = v__42.x;
    conv2d_nchw[v__41.y] = v__42.y;
    conv2d_nchw[v__41.z] = v__42.z;
    conv2d_nchw[v__41.w] = v__42.w;
    int4 v__43 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602112))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602112))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602112))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602112))+(12544*3));
    half4 v__44 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1920));
    conv2d_nchw[v__43.x] = v__44.x;
    conv2d_nchw[v__43.y] = v__44.y;
    conv2d_nchw[v__43.z] = v__44.z;
    conv2d_nchw[v__43.w] = v__44.w;
    int4 v__45 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 32))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 32))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 32))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 32))+(12544*3));
    half4 v__46 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2176));
    conv2d_nchw[v__45.x] = v__46.x;
    conv2d_nchw[v__45.y] = v__46.y;
    conv2d_nchw[v__45.z] = v__46.z;
    conv2d_nchw[v__45.w] = v__46.w;
    int4 v__47 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200736))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200736))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200736))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200736))+(12544*3));
    half4 v__48 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2432));
    conv2d_nchw[v__47.x] = v__48.x;
    conv2d_nchw[v__47.y] = v__48.y;
    conv2d_nchw[v__47.z] = v__48.z;
    conv2d_nchw[v__47.w] = v__48.w;
    int4 v__49 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401440))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401440))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401440))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401440))+(12544*3));
    half4 v__50 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2688));
    conv2d_nchw[v__49.x] = v__50.x;
    conv2d_nchw[v__49.y] = v__50.y;
    conv2d_nchw[v__49.z] = v__50.z;
    conv2d_nchw[v__49.w] = v__50.w;
    int4 v__51 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602144))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602144))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602144))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602144))+(12544*3));
    half4 v__52 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2944));
    conv2d_nchw[v__51.x] = v__52.x;
    conv2d_nchw[v__51.y] = v__52.y;
    conv2d_nchw[v__51.z] = v__52.z;
    conv2d_nchw[v__51.w] = v__52.w;
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1152)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[4], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1408)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[5], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1664)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[6], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[((((int)threadIdx.y) * 1024) + 1920)])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[7], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
    int4 v__53 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 16))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 16))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 16))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 16))+(12544*3));
    half4 v__54 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1152));
    conv2d_nchw[v__53.x] = v__54.x;
    conv2d_nchw[v__53.y] = v__54.y;
    conv2d_nchw[v__53.z] = v__54.z;
    conv2d_nchw[v__53.w] = v__54.w;
    int4 v__55 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200720))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200720))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200720))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200720))+(12544*3));
    half4 v__56 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1408));
    conv2d_nchw[v__55.x] = v__56.x;
    conv2d_nchw[v__55.y] = v__56.y;
    conv2d_nchw[v__55.z] = v__56.z;
    conv2d_nchw[v__55.w] = v__56.w;
    int4 v__57 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401424))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401424))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401424))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401424))+(12544*3));
    half4 v__58 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1664));
    conv2d_nchw[v__57.x] = v__58.x;
    conv2d_nchw[v__57.y] = v__58.y;
    conv2d_nchw[v__57.z] = v__58.z;
    conv2d_nchw[v__57.w] = v__58.w;
    int4 v__59 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602128))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602128))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602128))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602128))+(12544*3));
    half4 v__60 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 1920));
    conv2d_nchw[v__59.x] = v__60.x;
    conv2d_nchw[v__59.y] = v__60.y;
    conv2d_nchw[v__59.z] = v__60.z;
    conv2d_nchw[v__59.w] = v__60.w;
    int4 v__61 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 48))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 48))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 48))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 48))+(12544*3));
    half4 v__62 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2176));
    conv2d_nchw[v__61.x] = v__62.x;
    conv2d_nchw[v__61.y] = v__62.y;
    conv2d_nchw[v__61.z] = v__62.z;
    conv2d_nchw[v__61.w] = v__62.w;
    int4 v__63 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200752))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200752))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200752))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 200752))+(12544*3));
    half4 v__64 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2432));
    conv2d_nchw[v__63.x] = v__64.x;
    conv2d_nchw[v__63.y] = v__64.y;
    conv2d_nchw[v__63.z] = v__64.z;
    conv2d_nchw[v__63.w] = v__64.w;
    int4 v__65 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401456))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401456))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401456))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 401456))+(12544*3));
    half4 v__66 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2688));
    conv2d_nchw[v__65.x] = v__66.x;
    conv2d_nchw[v__65.y] = v__66.y;
    conv2d_nchw[v__65.z] = v__66.z;
    conv2d_nchw[v__65.w] = v__66.w;
    int4 v__67 = make_int4(((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602160))+(12544*0), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602160))+(12544*1), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602160))+(12544*2), ((((((((((int)threadIdx.x) & 3) * 50176) + (((int)blockIdx.y) * 448)) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.y) * 8)) + (((int)threadIdx.x) >> 2)) + 602160))+(12544*3));
    half4 v__68 = *(half4*)(((half*)buf_dyn_shmem) + (((((int)threadIdx.y) * 128) + (((int)threadIdx.x) * 4)) + 2944));
    conv2d_nchw[v__67.x] = v__68.x;
    conv2d_nchw[v__67.y] = v__68.y;
    conv2d_nchw[v__67.z] = v__68.z;
    conv2d_nchw[v__67.w] = v__68.w;
}

