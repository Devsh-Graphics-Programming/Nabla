#ifndef __I_ELEMENT_FACTORY_H_INCLUDED__
#define __I_ELEMENT_FACTORY_H_INCLUDED__


namespace irr
{
namespace ext
{
namespace MitsubaLoader
{

class IElement;

class CElementFactory
{
public:
	//constructs certain elements based on element's name and its attributes
	static IElement* createElement(const char* _el, const char** _atts);

private:
	static IElement* parseScene(const char* _el, const char** _atts);
	static IElement* parseShape(const char* _el, const char** _atts);

};

}
}
}

#endif