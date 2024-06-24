import json
from re import search

emptyline = f""

class ContinueEx(Exception):
    pass

def loadJSON(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as ex:
        print(f"Error while reading file: {file_path}\nException: {ex}")
        raise ex

def formatValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value

def buildComment(comment, res, sectionName):
    for commentLine in comment['comment']:
        res.append(f"    // {commentLine}")
    if "entries" in comment:
        for entry in comment['entries']:
            buildVariable(entry, res, sectionName)

def buildVariable(variable, res, sectionName):
    if "comment" in variable:
        res.append("    // " + variable['comment'])

    expose = variable["expose"] if "expose" in variable else "DEFAULT"
    commentDeclaration = "// " if expose != "DEFAULT" else ""
    constexprDeclaration = "constexpr static inline " if sectionName == "constexprs" else ""
    valueDeclaration = f" = {formatValue(variable['value'])}" if variable['value'] != None else ""
    line = f"    {commentDeclaration}{constexprDeclaration}{variable['type']} {variable['name']}{valueDeclaration};"
    res.append(line)

def buildDeviceHeader(**params):
    res = []

    for sectionName, sectionContent in params['json'].items():
        for dict in sectionContent:
            if "type" in dict:
                buildVariable(dict, res, sectionName)
            else:
                buildComment(dict, res, sectionName)
            res.append(emptyline)

    return res

def SubsetMethodHelper(dict, res):
    # its in gen.py
    expose = "DISABLE" if "expose" in dict else "DEFAULT"
    if 'compareSkip' in dict and dict['compareSkip'] or expose == "DISABLE":
        return
    line = "    "
    if 'compareExpose' in dict and not dict['compareExpose']:
        line += "// "
    numeric_types = ['uint8_t', 'uint16_t', 'uint32_t', 'uint64_t', 'size_t', 'int8_t', 'int32_t', 'float', 'asset::CGLSLCompiler::E_SPIRV_VERSION']
    if dict['type'] in numeric_types:
        is_array = "[" in dict['name']
        array_size = int(search(r'\[\d+\]', dict['name']).group(0)[1:-1]) if is_array else 0
        is_range = "Range" in dict['name']
        if is_array:
            array_size_count = len(str(array_size))
            array_name = dict['name'][:-(array_size_count + 2)]
            if is_range:
                line += f"if ({array_name}[0] < _rhs.{array_name}[0] || {array_name}[1] > _rhs.{array_name}[1]) return false;"
            else:
                lines = [line] * array_size
                for i in range(array_size):
                    lines[i] += f"if ({array_name}[{i}] > _rhs.{array_name}[{i}]) return false;"
                line = "\n".join(lines)
        else:
            signDeclaration = "<" if 'compareFlipped' in dict and dict['compareFlipped'] else ">"
            line += f"if ({dict['name']} {signDeclaration} _rhs.{dict['name']}) return false;"
    elif dict['type'].startswith("core::bitflag<"):
        line += f"if (!_rhs.{dict['name']}.hasFlags({dict['name']})) return false;"
    elif dict['type'] == "bool":
        line += f"if ({dict['name']} && !_rhs.{dict['name']}) return false;"
    elif dict['type'] == "E_POINT_CLIPPING_BEHAVIOR":
        line += f"if ({dict['name']}==EPCB_ALL_CLIP_PLANES && _rhs.{dict['name']}==EPCB_USER_CLIP_PLANES_ONLY) return false;"
    elif dict['type'].startswith("hlsl"):
        lines = [line, line]
        for i in range(2):
            componentDeclaration = ".x" if (i == 0) else ".y"
            signDeclaration = "<" if 'compareFlipped' in dict and dict['compareFlipped'] else ">"
            lines[i] = f"if ({dict['name']}{componentDeclaration} {signDeclaration} _rhs.{dict['name']}{componentDeclaration}) return false;"
        line = "\n".join(lines)

    res.append(line)


def buildSubsetMethod(**params):
    res = []

    for sectionName, sectionContent in params['json'].items():
        if sectionName == "constexprs":
            continue
        for dict in sectionContent:
            if 'type' in dict:
                SubsetMethodHelper(dict, res)

    return res

def transformFeaturesMethod(dict, op):
    expose = dict['expose'] if "expose" in dict else "DEFAULT"
    if expose != "DEFAULT":
        raise ContinueEx
    return f"    res.{dict['name']} {op}= _rhs.{dict['name']};"


def buildFeaturesMethod(**params):
    res = []

    sectionHeaders = {
        "vulkan10core": "VK 1.0 Core",
        "vulkan11core": "VK 1.1 Everything is either a Limit or Required",
        "vulkan12core": "VK 1.2",
        "vulkan13core": "VK 1.3",
        "nablacore": "Nabla Core Extensions",
        "vulkanext": "Extensions"
    }

    for sectionName, sectionContent in params['json'].items():
        if sectionName == "nabla":
            continue
        res.append(f"   // {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            if 'type' in dict:
                try:
                    res.append(transformFeaturesMethod(dict, params['op']))
                except ContinueEx:
                    continue
            if "entries" in dict:
                for entry in dict['entries']:
                    try:
                        res.append(transformFeaturesMethod(entry, params['op']))
                    except ContinueEx:
                        continue

    return res

def buildTraitsHeaderHelper(res, name, json_data, line_format, *line_format_params):
    sectionHeaders = {
        "vulkan10core": "VK 1.0",
        "vulkan11core": "VK 1.1",
        "vulkan12core": "VK 1.2",
        "vulkan13core": "VK 1.3",
        "nablacore": "Nabla Core Extensions",
        "vulkanext": "Extensions",
        "nabla": "Nabla"
    }

    res.append(f"// {name}")
    for sectionName, sectionContent in json_data.items():
        if sectionName == "constexprs":
            continue
        if sectionName in sectionHeaders:
            res.append(f"// {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            if 'type' in dict:
                try:
                    expose = "expose" in dict and dict["expose"] or "expose" not in dict
                    if not expose:
                        continue

                    for param in line_format_params:
                        if param not in dict:
                            raise ContinueEx

                    line = line_format.format(*[formatValue(dict[param]) for param in line_format_params])
                    res.append(line)
                except ContinueEx:
                    continue

def buildJITTraitsHeaderHelper(res, name, json_data, json_type, *line_format_params):
    sectionHeaders = {
        "vulkan10core": "VK 1.0",
        "vulkan11core": "VK 1.1",
        "vulkan12core": "VK 1.2",
        "vulkan13core": "VK 1.3",
        "nablacore": "Nabla Core Extensions",
        "vulkanext": "Extensions",
        "nabla": "Nabla"
    }

    res.append(f"// {name}")
    for sectionName, sectionContent in json_data.items():
        if sectionName == "constexprs":
            continue
        if sectionName in sectionHeaders:
            res.append(f"// {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            line_format="NBL_CONSTEXPR_STATIC_INLINE {} {} = )===\" + CJITIncludeLoader::to_string({}.{}) + R\"===(;"
            if 'type' in dict:
                try:
                    dict["json_type"] = json_type 
                    expose = "expose" in dict and dict["expose"] or "expose" not in dict
                    if not expose:
                        continue

                    for param in line_format_params:
                        if param not in dict:
                            raise ContinueEx
                    line = line_format.format(*[formatValue(dict[param]) for param in line_format_params])
                    res.append(line)
                except ContinueEx:
                    continue


def buildTraitsHeader(**params):
    res = []

    buildTraitsHeaderHelper(
        res,
        f"Limits {params['type']}",
        params["limits_json"],
        params["template"],
        *params['format_params']
    )
    res.append(emptyline)
    buildTraitsHeaderHelper(
        res,
        f"Features {params['type']}",
        params["features_json"],
        params["template"],
        *params['format_params']
    )

    return res

def buildJITTraitsHeader(**params):
    res = [
        "std::string jit_traits = R\"===("
    ]

    buildJITTraitsHeaderHelper(
        res,
        f"Limits {params['type']}",
        params["limits_json"],
        "limits",
        *params['format_params']
    )
    res.append(emptyline)
    buildJITTraitsHeaderHelper(
        res,
        f"Features {params['type']}",
        params["features_json"],
        "features",
        *params['format_params']
    )

    res.append(")===\";")

    return res

def writeHeader(file_path, header_builder, **params):
    try:
        with open(file_path, mode="w") as file:
            device_header = header_builder(**params)
            for line in device_header:
                file.write(line + '\n')
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")
        raise ex