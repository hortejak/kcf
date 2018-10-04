#include "debug.h"
#include <string>

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<cv::Mat> &p)
{
    IOSave s(os);
    os << std::setprecision(DbgTracer::precision);
    os << p.obj.size << " " << p.obj.channels() << "ch ";// << static_cast<const void *>(p.obj.data);
    os << " = [ ";
    constexpr size_t num = 10;
    for (size_t i = 0; i < std::min(num, p.obj.total()); ++i)
        os << p.obj.ptr<float>()[i] << ", ";
    os << (num < p.obj.total() ? "... ]" : "]");
    return os;
}

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<ComplexMat> &p)
{
    IOSave s(os);
    os << std::setprecision(DbgTracer::precision);
    os << "<cplx> " << p.obj.size() << " " << p.obj.channels() << "ch "; // << p.obj.get_p_data();
    constexpr int num = 10;
    for (uint s = 0; s < p.obj.n_scales; ++s) {
        uint ofs = s * p.obj.rows * p.obj.cols * p.obj.n_channels / p.obj.n_scales;
        os << " = [ ";
        for (int i = 0; i < std::min(num, p.obj.size().area()); ++i) {
            std::complex<float> c = p.obj.get_p_data()[ofs + i];
            if (fabs(c.real()) < 5e-7)
                c = std::complex<float>(0, c.imag());
            if (fabs(c.imag()) < 5e-7)
                c = std::complex<float>(c.real(), 0);
            os << c << ", ";
        }
        os << (num < p.obj.size().area() ? "... ]" : "]");
    }
    return os;
}
