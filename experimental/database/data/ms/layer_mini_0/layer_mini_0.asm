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
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
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
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ A, half* __restrict__ W, half* __restrict__ conv2d_nchw);
extern "C" __global__ void __launch_bounds__(32) main_kernel(half* __restrict__ A, half* __restrict__ W, half* __restrict__ conv2d_nchw) {
  extern __shared__ uchar buf_dyn_shmem[];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> conv2d_nchw_reindex_shared_dyn_wmma_accumulator[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> W_reindex_pad_shared_dyn_wmma_matrix_b[2];
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], 0.000000e+00f);
  nvcuda::wmma::fill_fragment(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], 0.000000e+00f);
  half condval;
  if (((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + ((((int)threadIdx.x) & 15) % 7))))) {
    condval = A[(((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval;
  half condval_1;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_1 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_1 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_1;
  half condval_2;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_2 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_2 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_2;
  half condval_3;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_3 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_3 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_3;
  half condval_4;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_4 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_4 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_4;
  half condval_5;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_5 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_5 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_5;
  half condval_6;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_6 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_6 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_6;
  half condval_7;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_7 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 675)];
  } else {
    condval_7 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_7;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2))];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 588)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 589)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1176)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1177)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1764)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1765)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2352)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2353)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2940)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2941)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3528)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3529)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4116)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4117)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_8;
  if (((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7))))) {
    condval_8 = A[(((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_8 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_8;
  half condval_9;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_9 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_9 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_9;
  half condval_10;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_10 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_10 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_10;
  half condval_11;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_11 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_11 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_11;
  half condval_12;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_12 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_12 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_12;
  half condval_13;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_13 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_13 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_13;
  half condval_14;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_14 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_14 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_14;
  half condval_15;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 16) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_15 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 16) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_15 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_15;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 16)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 17)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 604)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 605)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1192)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1193)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1780)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1781)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2368)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2369)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2956)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2957)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3544)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3545)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4132)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4133)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_16;
  if ((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7))))) {
    condval_16 = A[(((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_16 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_16;
  half condval_17;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_17 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_17 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_17;
  half condval_18;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_18 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_18 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_18;
  half condval_19;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_19 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_19 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_19;
  half condval_20;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_20 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_20 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_20;
  half condval_21;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_21 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_21 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_21;
  half condval_22;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_22 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_22 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_22;
  half condval_23;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 32) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_23 = A[((((((((int)blockIdx.y) * 3136) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 32) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7)) - 675)];
  } else {
    condval_23 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_23;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 32)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 33)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 620)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 621)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1208)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1209)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1796)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1797)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2384)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2385)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2972)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2973)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3560)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3561)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4148)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4149)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_24;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7))))) {
    condval_24 = A[(((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_24 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_24;
  half condval_25;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_25 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_25 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_25;
  half condval_26;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_26 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_26 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_26;
  half condval_27;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_27 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_27 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_27;
  half condval_28;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_28 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_28 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_28;
  half condval_29;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_29 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_29 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_29;
  half condval_30;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_30 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_30 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_30;
  half condval_31;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 48) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 6) % 7)) < 227))) {
    condval_31 = A[((((((((((((int)threadIdx.x) & 15) + 48) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 48) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 6) % 7)) - 675)];
  } else {
    condval_31 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_31;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 48)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 49)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 636)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 637)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1224)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1225)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1812)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1813)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2400)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2401)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2988)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2989)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3576)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3577)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4164)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4165)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_32;
  if (((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7))))) {
    condval_32 = A[(((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_32 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_32;
  half condval_33;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_33 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_33 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_33;
  half condval_34;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_34 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_34 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_34;
  half condval_35;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_35 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_35 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_35;
  half condval_36;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_36 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_36 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_36;
  half condval_37;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_37 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_37 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_37;
  half condval_38;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_38 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_38 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_38;
  half condval_39;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 1) % 7)) < 227))) {
    condval_39 = A[((((((((((((int)threadIdx.x) & 15) + 64) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 1) % 7)) - 675)];
  } else {
    condval_39 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_39;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 64)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 65)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 652)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 653)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1240)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1241)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1828)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1829)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2416)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2417)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3004)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3005)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3592)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3593)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4180)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4181)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_40;
  if ((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7))))) {
    condval_40 = A[(((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_40 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_40;
  half condval_41;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_41 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_41 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_41;
  half condval_42;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_42 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_42 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_42;
  half condval_43;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_43 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_43 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_43;
  half condval_44;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_44 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_44 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_44;
  half condval_45;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_45 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_45 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_45;
  half condval_46;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_46 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_46 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_46;
  half condval_47;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 31) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 3) % 7)) < 227))) {
    condval_47 = A[((((((((((((int)threadIdx.x) & 15) + 80) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 31) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 3) % 7)) - 675)];
  } else {
    condval_47 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_47;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 80)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 81)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 668)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 669)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1256)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1257)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1844)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1845)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2432)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2433)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3020)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3021)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3608)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3609)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4196)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4197)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_48;
  if ((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7))))) {
    condval_48 = A[(((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_48 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_48;
  half condval_49;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_49 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_49 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_49;
  half condval_50;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_50 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_50 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_50;
  half condval_51;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_51 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_51 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_51;
  half condval_52;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_52 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_52 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_52;
  half condval_53;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_53 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_53 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_53;
  half condval_54;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_54 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_54 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_54;
  half condval_55;
  if (((((3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 47) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 5) % 7)) < 227))) {
    condval_55 = A[((((((((((((int)threadIdx.x) & 15) + 96) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((((int)threadIdx.x) & 15) + 47) % 49) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 5) % 7)) - 675)];
  } else {
    condval_55 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_55;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 96)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 97)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 684)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 685)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1272)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1273)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1860)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1861)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2448)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2449)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3036)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3037)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3624)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3625)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4212)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4213)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_56;
  if (((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + ((((int)threadIdx.x) & 15) % 7))))) {
    condval_56 = A[(((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_56 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_56;
  half condval_57;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_57 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_57 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_57;
  half condval_58;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_58 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_58 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_58;
  half condval_59;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_59 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_59 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_59;
  half condval_60;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_60 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_60 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_60;
  half condval_61;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_61 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_61 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_61;
  half condval_62;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_62 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_62 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_62;
  half condval_63;
  if ((((1 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((int)threadIdx.x) & 15) / 7))) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + ((((int)threadIdx.x) & 15) % 7)) < 227))) {
    condval_63 = A[((((((((((((int)threadIdx.x) & 15) + 112) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + (((((int)threadIdx.x) & 15) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + ((((int)threadIdx.x) & 15) % 7)) - 227)];
  } else {
    condval_63 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_63;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 112)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 113)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 700)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 701)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1288)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1289)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1876)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1877)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2464)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2465)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3052)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3053)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3640)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3641)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4228)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4229)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_64;
  if ((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7))))) {
    condval_64 = A[(((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_64 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_64;
  half condval_65;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_65 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_65 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_65;
  half condval_66;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_66 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_66 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_66;
  half condval_67;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_67 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_67 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_67;
  half condval_68;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_68 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_68 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_68;
  half condval_69;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_69 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_69 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_69;
  half condval_70;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_70 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_70 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_70;
  half condval_71;
  if (((((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + (((((int)threadIdx.x) & 15) + 30) / 7)) < 227) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 2) % 7)) < 227))) {
    condval_71 = A[((((((((((((int)threadIdx.x) & 15) + 128) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 30) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((((int)threadIdx.x) & 15) + 2) % 7)) - 675)];
  } else {
    condval_71 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_71;
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 128)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 129)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 716)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 717)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1304)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1305)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1892)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1893)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2480)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2481)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3068)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3069)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3656)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3657)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4244)];
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4245)];
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  half condval_72;
  if ((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((int)blockIdx.x) % 14) >> 1) * 32) + ((((int)threadIdx.x) >> 4) * 2)) + (((((int)threadIdx.x) & 15) + 4) % 7))))) {
    condval_72 = A[(((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + (((((int)blockIdx.x) % 14) >> 1) * 32)) + ((((int)threadIdx.x) >> 4) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_72 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[(((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15))] = condval_72;
  half condval_73;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_73 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 2) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_73 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 80)] = condval_73;
  half condval_74;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_74 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 4) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_74 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 160)] = condval_74;
  half condval_75;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_75 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 6) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_75 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 240)] = condval_75;
  half condval_76;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_76 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 8) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_76 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 320)] = condval_76;
  half condval_77;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_77 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 10) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_77 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 400)] = condval_77;
  half condval_78;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_78 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 12) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_78 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 480)] = condval_78;
  half condval_79;
  if (((((((((int)threadIdx.x) & 15) < 3) && (3 <= (((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)))) && ((((((int)blockIdx.y) * 14) + ((((int)blockIdx.x) / 14) * 2)) + ((((((int)threadIdx.x) & 15) + 46) % 49) / 7)) < 227)) && (3 <= (((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)))) && ((((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2) + (((((int)threadIdx.x) & 15) + 4) % 7)) < 227))) {
    condval_79 = A[((((((((((((int)threadIdx.x) & 15) + 144) / 49) * 50176) + (((int)blockIdx.y) * 3136)) + ((((int)blockIdx.x) / 14) * 448)) + ((((((int)threadIdx.x) & 15) + 46) / 7) * 224)) + ((((((((int)blockIdx.x) >> 1) * 16) + (((int)threadIdx.x) >> 4)) + 14) % 112) * 2)) + (((int)threadIdx.x) & 15)) - 671)];
  } else {
    condval_79 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 4) * 40) + (((int)threadIdx.x) & 15)) + 560)] = condval_79;
  half condval_80;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_80 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 144)];
  } else {
    condval_80 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 640)] = condval_80;
  half condval_81;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_81 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 145)];
  } else {
    condval_81 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 641)] = condval_81;
  half condval_82;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_82 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 732)];
  } else {
    condval_82 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 800)] = condval_82;
  half condval_83;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_83 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 733)];
  } else {
    condval_83 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 801)] = condval_83;
  half condval_84;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_84 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1320)];
  } else {
    condval_84 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 960)] = condval_84;
  half condval_85;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_85 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1321)];
  } else {
    condval_85 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 961)] = condval_85;
  half condval_86;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_86 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1908)];
  } else {
    condval_86 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1120)] = condval_86;
  half condval_87;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_87 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 1909)];
  } else {
    condval_87 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1121)] = condval_87;
  half condval_88;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_88 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2496)];
  } else {
    condval_88 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1280)] = condval_88;
  half condval_89;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_89 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 2497)];
  } else {
    condval_89 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1281)] = condval_89;
  half condval_90;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_90 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3084)];
  } else {
    condval_90 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1440)] = condval_90;
  half condval_91;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_91 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3085)];
  } else {
    condval_91 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1441)] = condval_91;
  half condval_92;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_92 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3672)];
  } else {
    condval_92 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1600)] = condval_92;
  half condval_93;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_93 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 3673)];
  } else {
    condval_93 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1601)] = condval_93;
  half condval_94;
  if (((((int)threadIdx.x) & 7) < 2)) {
    condval_94 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4260)];
  } else {
    condval_94 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1760)] = condval_94;
  half condval_95;
  if (((((int)threadIdx.x) & 7) < 1)) {
    condval_95 = W[(((((((int)blockIdx.x) & 1) * 4704) + ((((int)threadIdx.x) >> 3) * 147)) + ((((int)threadIdx.x) & 7) * 2)) + 4261)];
  } else {
    condval_95 = __float2half_rn(0.000000e+00f);
  }
  ((half*)buf_dyn_shmem)[((((((int)threadIdx.x) >> 3) * 40) + ((((int)threadIdx.x) & 7) * 2)) + 1761)] = condval_95;
  __syncthreads();
  nvcuda::wmma::load_matrix_sync(pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], (&(((half*)buf_dyn_shmem)[0])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[0], (&(((half*)buf_dyn_shmem)[640])), 40);
  nvcuda::wmma::load_matrix_sync(W_reindex_pad_shared_dyn_wmma_matrix_b[1], (&(((half*)buf_dyn_shmem)[1280])), 40);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[0], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0]);
  nvcuda::wmma::mma_sync(conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], pad_temp_reindex_pad_shared_dyn_wmma_matrix_a[0], W_reindex_pad_shared_dyn_wmma_matrix_b[1], conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1]);
  __syncthreads();
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[0])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[0], 16, nvcuda::wmma::mem_row_major);
  nvcuda::wmma::store_matrix_sync((&(((half*)buf_dyn_shmem)[256])), conv2d_nchw_reindex_shared_dyn_wmma_accumulator[1], 16, nvcuda::wmma::mem_row_major);
  __syncthreads();
    int2 v_ = make_int2((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)))+(12544*0), (((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)))+(12544*1));
    half2 v__1 = *(half2*)(((half*)buf_dyn_shmem) + (((int)threadIdx.x) * 2));
    conv2d_nchw[v_.x] = v__1.x;
    conv2d_nchw[v_.y] = v__1.y;
    int2 v__2 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 4))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 4))+(12544*1));
    half2 v__3 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 64));
    conv2d_nchw[v__2.x] = v__3.x;
    conv2d_nchw[v__2.y] = v__3.y;
    int2 v__4 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 8))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 8))+(12544*1));
    half2 v__5 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 128));
    conv2d_nchw[v__4.x] = v__5.x;
    conv2d_nchw[v__4.y] = v__5.y;
    int2 v__6 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 12))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 12))+(12544*1));
    half2 v__7 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 192));
    conv2d_nchw[v__6.x] = v__7.x;
    conv2d_nchw[v__6.y] = v__7.y;
    int2 v__8 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200704))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200704))+(12544*1));
    half2 v__9 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 256));
    conv2d_nchw[v__8.x] = v__9.x;
    conv2d_nchw[v__8.y] = v__9.y;
    int2 v__10 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200708))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200708))+(12544*1));
    half2 v__11 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 320));
    conv2d_nchw[v__10.x] = v__11.x;
    conv2d_nchw[v__10.y] = v__11.y;
    int2 v__12 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200712))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200712))+(12544*1));
    half2 v__13 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 384));
    conv2d_nchw[v__12.x] = v__13.x;
    conv2d_nchw[v__12.y] = v__13.y;
    int2 v__14 = make_int2(((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200716))+(12544*0), ((((((((((int)blockIdx.x) & 1) * 401408) + ((((int)threadIdx.x) & 7) * 25088)) + (((int)blockIdx.y) * 784)) + ((((int)blockIdx.x) >> 1) * 16)) + (((int)threadIdx.x) >> 3)) + 200716))+(12544*1));
    half2 v__15 = *(half2*)(((half*)buf_dyn_shmem) + ((((int)threadIdx.x) * 2) + 448));
    conv2d_nchw[v__14.x] = v__15.x;
    conv2d_nchw[v__14.y] = v__15.y;
}

