import json
from re import search

emptyline = f""

def loadJSON(file_path):
    try:
        with open(file_path) as file:
            return json.load(file)
    except Exception as ex:
        print(f"Error while reading file: {file_path}\nException: {ex}")

def buildComment(comment, res):
    res.append("    /*")
    for commentLine in comment['groupComment']:
        res.append(f"       {commentLine}")
    res.append("    */")

def buildStatement(statement, res):
    res.append(f"   {statement['statement']}")

def buildFunction(function, res):
    for functionLine in function["function"]:
        res.append(f"   {functionLine}")
    res.append(emptyline)

def buildEnum(dict, res):
    res.append("    enum " + dict['name'])
    res.append("    {")
    for declaration in dict['declarations']:
        res.append("    " + declaration + ",")
    res.append("    };")
    res.append(emptyline)

def formatValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value

def buildVariable(variable, res, sectionName):
    declarationOnly = "declare" in variable and variable["declare"]
    expose = "expose" in variable and variable["expose"] or "expose" not in variable
    commentDeclaration = "// " if not expose else ""
    constexprDeclaration = "constexpr static inline " if sectionName == "constexprs" else ""
    valueDeclaration = f" = {formatValue(variable['value'])}" if not declarationOnly else ""
    trailingCommentDeclaration = f" // {variable['comment']}" if "comment" in variable else ""
    line = f"    {commentDeclaration}{constexprDeclaration}{variable['type']} {variable['name']}{valueDeclaration};{trailingCommentDeclaration}"
    res.append(line)

def buildDeviceHeader(device_json):
    res = []

    for sectionName, sectionContent in device_json.items():
        for dict in sectionContent:
            if 'groupComment' in dict:
                buildComment(dict, res)
            elif "statement" in dict:
                buildStatement(dict, res)
            elif dict['type'] == "function":
                buildFunction(dict, res)
            elif dict['type'] == "enum":
                buildEnum(dict, res)
            else:
                buildVariable(dict, res, sectionName)

    return res

def SubsetMethodHelper(dict, res):
    expose = "expose" in dict and dict["expose"] or "expose" not in dict
    if 'compareSkip' in dict and dict['compareSkip'] or not expose:
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


def buildSubsetMethod(device_json):
    res = []

    for sectionName, sectionContent in device_json.items():
        if sectionName == "constexprs":
            continue
        for dict in sectionContent:
            if 'type' in dict:
                SubsetMethodHelper(dict, res)

    res.append(emptyline)
    res.append("    return true;")

    return res

def writeDeviceHeader(file_path, device_json):
    try:
        with open(file_path, mode="w") as file:
            device_header = buildDeviceHeader(device_json)
            for line in device_header:
                file.write(line + '\n')
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")

def writeSubsetMethod(file_path, device_json):
    try:
        with open(file_path, mode="w") as file:
            limits_method = buildSubsetMethod(device_json)
            print(limits_method)
            for line in limits_method:
                file.write(line + '\n')
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")
