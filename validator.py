import os
import pandas as pd
from requests import post



def Validate(folder_path, email, password, url="https://www.itc2019.org/itc2019-validator", timeout=30):
    """
    Send the XML file as the request body to the validator using HTTP Basic Auth.
    Returns the requests.Response object.
    """
    with open(folder_path, "rb") as f:
        data = f.read()

    headers = {"Content-Type": "text/xml;charset=UTF-8"}
    resp = post(url, auth=(email, password), headers=headers, data=data, timeout=timeout)
    return resp

def report_result(file):
    url = "https://qngpumipdgtk.sealosbja.site/api/report"
    header = {"token": "15828030238"}
    email = "scxsz1@nottingham.edu.cn"
    password = "#Zhsh15828030238"
    response = Validate(file, email, password)
    print("Status Code:", response.status_code, end=" ")
    if response.status_code == 200:
        result = response.json()
        # print("Validation Result:", result)
        assignedVariables = result.get("assignedVariables")
        if assignedVariables['percent'] == 100.0:
            data = {
                "instance": result.get("instance", "error"),
                "valid": "valid",
                "Total cost": result.get("totalCost", {}).get('value', -1),
                "Time penalty": result.get("timePenalty", {}).get('value', -1),
                "Room penalty": result.get("roomPenalty", {}).get('value', -1),
                "Distribution penalty": result.get("distributionPenalty", {}).get('value', -1),
                "Student conflicts": result.get("studentConflicts", {}).get('value', -1),
                "Runtime": result.get("runtime", -1),
                "Cores": result.get("cores", -1),
                "Technique": result.get("technique", "error")
            }
        else:
            data = {
                "instance": result.get("instance", "error"),
                "valid": "invalid",
                "Total cost": result.get("totalCost", {}).get('value', -1),
                "Time penalty": result.get("timePenalty", {}).get('value', -1),
                "Room penalty": result.get("roomPenalty", {}).get('value', -1),
                "Distribution penalty": result.get("distributionPenalty", {}).get('value', -1),
                "Student conflicts": result.get("studentConflicts", {}).get('value', -1),
                "Runtime": result.get("runtime", -1),
                "Cores": result.get("cores", -1),
                "Technique": result.get("technique", "error")
            }
        resp = post(url, headers=header, json=data)
        print("Report Status Code:", resp.status_code)
        return data
    else:
        print("Validation failed.", response.status_code)
        print("Response Text:\n", response.text)
        return None
        # print("Report Response Text:\n", resp.text)

if __name__ == "__main__":
    file = '/home/scxsz1/zsh/Learning/MARL/PSTT/MARL/MAPPO/position/results/muni-fi-fal17.last_solution.xml'
    data = report_result(file)
    print(data)
    