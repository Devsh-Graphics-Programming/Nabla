import json
import os

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

def formatValue(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value

def buildVariable(variable, res, sectionName):
    declarationOnly = "declare" in variable and variable["declare"]
    expose = "expose" in variable and variable["expose"] or "expose" not in variable
    line = "    "
    if not expose:
        line += "// "
    if sectionName == "constexprs":
        line += "constexpr static inline "
    line += f"{variable['type']} {variable['name']}"
    if not declarationOnly:
        line += f" = {formatValue(variable['value'])}"
    line += ";"
    if "comment" in variable:
        line += f" // {variable['comment']}" 
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
            else:
                buildVariable(dict, res, sectionName)
                pass

    return res

def writeDeviceHeader(file_path, device_json):
    try:
        with open(file_path, mode="w") as file:
            device_header = buildDeviceHeader(device_json)
            for line in device_header:
                file.write(line + '\n')
    except Exception as ex:
        print(f"Error while writing to file: {file_path}\nException: {ex}")
