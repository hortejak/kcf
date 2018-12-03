#ifndef PREM_HPP
#define PREM_HPP

#define PREMIZABLE

#ifdef PREMIZABLE

constexpr int complexmat_h = 128;
#if defined(CUFFT) || defined(FFTW)
constexpr int complexmat_w = complexmat_h / 2 + 1;
#else
constexpr int complexmat_w = complexmat_h;
#endif

#define PREM_CONSTEXPR constexpr
#define PREM_STATIC_CONSTEXPR static constexpr
#define PREM(x) x
#define NON_PREM(x)

#else

#define PREM_CONSTEXPR
#define PREM_STATIC_CONSTEXPR
#define PREM(x)
#define NON_PREM(x) x

#endif

#endif // PREM_HPP
