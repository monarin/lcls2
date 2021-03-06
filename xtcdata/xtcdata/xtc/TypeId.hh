#ifndef XtcData_TypeId_hh
#define XtcData_TypeId_hh

#include <stdint.h>

namespace XtcData
{

class TypeId
{
public:
    /*
     * Notice: New enum values should be appended to the end of the enum list, since
     *   the old values have already been recorded in the existing xtc files.
     */
    enum Type { Parent, ShapesData, Shapes, Data, Names, NumberOf };

    TypeId()
    {
    }
    TypeId(const TypeId& v);
    TypeId(Type type, uint32_t version);
    TypeId(const char*);

    Type id() const;
    uint32_t version() const;
    uint32_t value() const;

    static const char* name(Type type);
    static uint32_t _sizeof()
    {
        return sizeof(TypeId);
    }

private:
    uint32_t _value;
};
}

#endif
