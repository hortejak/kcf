#include "debug.h"

std::ostream &operator<<(std::ostream &os, const DbgTracer::Printer<cv::Mat> &p)
{
    IOSave s(os);
    os << std::setprecision(3);
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
    os << std::setprecision(3);
    os << "<cplx> " << p.obj.size() << " " << p.obj.channels() << "ch "; // << p.obj.get_p_data();
    os << " = [ ";
    constexpr int num = 10;
    for (int i = 0; i < std::min(num, p.obj.size().area()); ++i)
        os << p.obj.get_p_data()[i] << ", ";
    os << (num < p.obj.size().area() ? "... ]" : "]");
    return os;
}
