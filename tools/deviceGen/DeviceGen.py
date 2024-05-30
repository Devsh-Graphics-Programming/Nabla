import json
import os

emptyline = f""

def buildPath(file_path):
    return os.path.abspath(os.path.join(os.path.dirname( __file__ ), file_path))

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

def formatValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value

def buildVariable(variable, res):
    declarationOnly = "declare" in variable and variable["declare"]
    expose = "expose" in variable and variable["expose"] or "expose" not in variable
    line = "    "
    if not expose:
        line += "// "
    line += f"{variable['type']} {variable['name']}"
    if not declarationOnly:
        line += f" = {formatValue(variable['value'])}"
    line += ";"
    if "comment" in variable:
        line += f" // {variable['comment']}" 
    res.append(line)

def buildDeviceHeader(device_json):
    res = []

    # Header Guard
    res.append(f"#ifndef {device_json['headerGuard']}")
    res.append(f"#define {device_json['headerGuard']}")
    res.append(emptyline)
    
    # Includes
    if 'includes' in device_json and 'includePath' in device_json:
        for include in device_json['includes']:
            res.append(f"#include \"{device_json['includePath']}{include}\"")
    if 'stlIncludes' in device_json:
        for include in device_json['stlIncludes']:
            res.append(f"#include <{include}>")
    res.append(emptyline)

    # Namespace
    res.append(f"namespace {device_json['namespace']}")
    res.append("{")
    res.append(emptyline)

    # Struct
    res.append("/*")
    for comment in device_json['structComment']:
        res.append(f"   {comment}")
    res.append("*/")
    res.append(f"struct {device_json['structName']}")
    res.append("{")

    # Content
    for _, sectionContent in device_json['content'].items():
        for dict in sectionContent:
            if 'groupComment' in dict:
                buildComment(dict, res)
            elif "statement" in dict:
                buildStatement(dict, res)
            elif dict['type'] == "function":
                buildFunction(dict, res)
            else:
                buildVariable(dict, res)
                pass

    # Close Struct
    res.append("};")
    res.append(emptyline)

    # Extra
    if "extra" in device_json:
        for line in device_json['extra']:
            res.append(line)
        res.append(emptyline)

    # Close Namespace
    res.append("}" + f" //{device_json['namespace']}")
    res.append(emptyline)

    # Close Header Guard
    res.append(f"#endif")

    return res

def writeDeviceHeader(file_path, device_json):
    try:
        with open(file_path, mode="w") as file:
            device_header = buildDeviceHeader(device_json)
            for line in device_header:
                file.write(line + '\n')
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")
