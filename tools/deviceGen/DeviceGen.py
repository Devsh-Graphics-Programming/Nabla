import json
from re import search, sub, split
from enum import IntFlag

emptyline = f""

class ContinueEx(Exception):
    pass

ExposeStatus = IntFlag("Expose", ["DEFAULT", "REQUIRE", "DISABLE", "MOVE_TO_LIMIT"])
CompareStatus = IntFlag("Compare", ["DEFAULT", "DISABLE", "SKIP", "REVERSE"])

MovedLimits = []

def computeStatus(status, string):
    return status(eval(sub(r"\w+", lambda s: f"{status[f"{s.group(0)}"]}", string)))

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
    temp_res = []
    require = False
    if "entries" in comment:
        for entry in comment['entries']:
            returnObj = buildVariable(entry, temp_res, sectionName, True)
            if returnObj and "require" in returnObj:
                require = require or returnObj["require"]

    for commentLine in comment['comment']:
        temp_res.insert(0, f"    // {"[REQUIRE]" if require else ""}{commentLine}")
    for line in temp_res:
        res.append(line)

def buildVariable(variable, res, sectionName, insideComment = False):
    expose = computeStatus(ExposeStatus, variable["expose"] if "expose" in variable else "DEFAULT")
    if expose == ExposeStatus.MOVE_TO_LIMIT:
        MovedLimits.append(variable)

    formattedValue = formatValue(variable['value'])
    exposeDeclaration = "// " if expose != ExposeStatus.DEFAULT else ""
    constexprDeclaration = "constexpr static inline " if sectionName == "constexprs" else ""
    valueDeclaration = f" = {formattedValue}" if variable['value'] != None else ""

    commentDeclaration = []
    returnObj = {}

    if expose == ExposeStatus.REQUIRE:
        if not insideComment:
            commentDeclaration.append("[REQUIRE]")
        else:
            returnObj['require'] = True
    if "comment" in variable:
        for comment in variable["comment"]:
            commentDeclaration.append(comment)

    for comment in commentDeclaration:
        res.append("    // " + comment)

    line = f"    {exposeDeclaration}{constexprDeclaration}{variable['type']} {variable['name']}{valueDeclaration};"
    res.append(line)

    return returnObj

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
    expose = computeStatus(ExposeStatus, dict['expose'] if "expose" in dict else "DEFAULT")
    compare = computeStatus(CompareStatus, dict['compare'] if "compare" in dict else "DEFAULT")
    if CompareStatus.SKIP in compare  or expose == ExposeStatus.DISABLE or expose == ExposeStatus.REQUIRE:
        return
    line = "    "
    if CompareStatus.DISABLE in compare:
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
            signDeclaration = "<" if CompareStatus.REVERSE in compare else ">"
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
            signDeclaration = "<" if CompareStatus.REVERSE in compare else ">"
            lines[i] = f"if ({dict['name']}{componentDeclaration} {signDeclaration} _rhs.{dict['name']}{componentDeclaration}) return false;"
        line = "\n".join(lines)

    res.append(line)

def buildSubsetMethod(**params):
    res = []

    sectionHeaders = {
        "vulkan10core": "VK 1.0 Core",
        "vulkan11core": "VK 1.1 Everything is either a Limit or Required",
        "vulkan12core": "VK 1.2",
        "vulkan13core": "VK 1.3",
        "nablacore": "Nabla Core Extensions",
        "core10": "Core 1.0",
        "vulkanext": "Extensions",
        "nabla": "Nabla"
    }

    for sectionName, sectionContent in params['json'].items():
        if sectionName == "constexprs":
            continue
        res.append(f"   // {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            if 'type' in dict:
                try:
                    SubsetMethodHelper(dict, res)
                except ContinueEx:
                    continue
            if 'entries' in dict:
                for entry in dict['entries']:
                    try:
                        SubsetMethodHelper(entry, res)
                    except ContinueEx:
                        continue
            res.append(emptyline)

    return res

def transformFeatures(dict, op):
    expose = computeStatus(ExposeStatus, dict['expose'] if "expose" in dict else "DEFAULT")
    if expose != ExposeStatus.DEFAULT:
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
        "vulkanext": "Extensions",
        "nabla": "Nabla"
    }

    for sectionName, sectionContent in params['json'].items():
        if sectionName == "nabla":
            continue
        res.append(f"   // {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            try:
                if 'type' in dict:
                    res.append(transformFeatures(dict, params['op']))
                    res.append(emptyline)
            except ContinueEx:
                continue
            if "entries" in dict:
                temp_res = []
                for entry in dict['entries']:
                    try:
                        temp_res.append(transformFeatures(entry, params['op']))
                    except ContinueEx:
                        continue
                if len(temp_res) > 0:
                    res.extend(temp_res)
                    res.append(emptyline)

    return res

def formatEnumType(type):
    type = type.split("::")[-1]
    type_parts = ''.join([c for c in type if c.isupper() or c == '_']).split('_')
    resultant_type = type_parts[1].lower()
    for type_part in type_parts[2:]:
        resultant_type += type_part.capitalize()
    return resultant_type

def formatEnumValue(type, value):
    value_parts = value.split(" ")
    temp_value_parts = []
    for value_part in value_parts:
        while (index := value_part.find("::")) != -1:
            value_part = value_part[index+2:]
        temp_value_parts.append(value_part)
    for i in range(len(temp_value_parts)):
        if temp_value_parts[i] == '|' or temp_value_parts[i] == '&':
            continue
        temp_value_parts[i] = type + "::" + temp_value_parts[i]
    return ' '.join(temp_value_parts)

def transformTraits(dict, line_format, json_type, line_format_params):
    expose = computeStatus(ExposeStatus, dict['expose'] if "expose" in dict else "DEFAULT")
    if expose != ExposeStatus.DEFAULT:
        raise ContinueEx
    if "type" not in dict or "value" not in dict or "name" not in dict:
        raise ContinueEx

    parsed_type = dict['type']
    parsed_name = dict['name']
    parsed_value = str(dict['value'])

    if parsed_type.startswith("core::bitflag") or parsed_type.startswith("asset"):
        resultant_type = formatEnumType(dict["type"])
        parsed_value = formatEnumValue(resultant_type, parsed_value)
        parsed_type = resultant_type

    if parsed_type.startswith("hlsl::"):
        index = parsed_type.find("_t")
        size = int(parsed_type[index + 2:])
        type_ext = parsed_type[6: index + 2]
        name_ext = [parsed_name + ext for ext in ["X", "Y", "Z", "W"]]
        cpp_name_ext = [parsed_name + ext for ext in [".x", ".y", ".z", ".w"]]
        value_ext = split(", |,", parsed_value[1:-2].strip())

        param_values = [
            {
                'type': type_ext,
                'name': name_ext[i],
                'cpp_name': cpp_name_ext[i],
                'value': value_ext[i],
                'json_type': json_type
            } for i in range(size)]

    elif (index1:= parsed_name.find('[')) != -1:
        is_range = parsed_name.find('Range') != -1
        index2 = parsed_name.find(']')
        size = int(parsed_name[index1 + 1: index2])
        type_ext = parsed_type
        name_ext = [parsed_name[:index1] + ext for ext in (["Min", "Max"] if is_range else ["X", "Y", "Z", ])]
        cpp_name_ext = [parsed_name[:index1] + ext for ext in ["[0]", "[1]", "[2]", "[3]"]]
        value_ext = split(", |,", parsed_value[1:-1].strip())

        param_values = [
            {
                'type': type_ext,
                'name': name_ext[i],
                'cpp_name': cpp_name_ext[i],
                'value': value_ext[i],
                'json_type': json_type
            } for i in range(size)]

    else:
        param_values = [
            {
                'type': parsed_type,
                'name': parsed_name,
                'cpp_name': parsed_name,
                'value': parsed_value,
                'json_type': json_type
            }
        ]

    return [line_format.format(*[formatValue(param_value[param]) for param in line_format_params]) for param_value in param_values]

def buildTraitsHeaderHelper(res, name, json_data, line_format, json_type, *line_format_params):
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
        # constexprs are handled specifically in buildTraitsHeader
        if sectionName == "constexprs":
            continue
        if sectionName in sectionHeaders:
            res.append(f"// {sectionHeaders[sectionName]}")
        for dict in sectionContent:
            if 'type' in dict:
                try:
                    for line in transformTraits(dict, line_format, json_type, line_format_params):
                        res.append(line)
                    res.append(emptyline)
                except ContinueEx:
                    continue
            if 'entries' in dict:
                temp_res = []
                for entry in dict['entries']:
                    try:
                        for line in transformTraits(entry, line_format, json_type, line_format_params):
                            temp_res.append(line)
                    except ContinueEx:
                        continue
                if len(temp_res) > 0:
                    res.extend(temp_res)
                    res.append(emptyline)

def buildTraitsHeader(**params):
    res = []

    if 'enable_constexprs' in params and params['enable_constexprs']:
        res.append('// constexprs')
        for entry in params["limits_json"]["constexprs"][0]["entries"]:
            expose = computeStatus(ExposeStatus, entry['expose'] if 'expose' in entry else "DEFAULT")
            if expose == ExposeStatus.DEFAULT:
                entry['type'] = sub("int8_t", "int16_t", entry['type'])
                res.append(f"NBL_CONSTEXPR_STATIC_INLINE {entry['type']} {entry['name']} = {entry['value']};")
        res.append(emptyline)

    if 'enable_jit' in params and params['enable_jit']:
        res.append("std::string jit_traits = R\"===(")

    buildTraitsHeaderHelper(
        res,
        f"Limits {params['type']}",
        params["limits_json"],
        params["template"],
        "limits",
        *params['format_params']
    )
    buildTraitsHeaderHelper(
        res,
        f"Features {params['type']}",
        params["features_json"],
        params["template"],
        "features",
        *params['format_params']
    )

    if 'enable_jit' in params and params['enable_jit']:
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