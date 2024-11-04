import json
from typing import List
from importlib.resources import files

def generate_art_techniques():

    scripts = """from cyberwheel.red_actions.technique import Technique
from cyberwheel.red_actions.atomic_test import AtomicTest
from typing import Any
"""
    preamble = "\ntechnique_mapping = {"
    metadata_path = files("cyberwheel.resources.metadata")
    path_to_combined_art_techniques = metadata_path.joinpath("combined_art_techniques.json")
    art_techniques = {}
    mapping = {}
    temp_list = []
    with open(path_to_combined_art_techniques, 'r') as f:
        art_techniques = json.load(f)
    for key in list(art_techniques.keys()):
        t = art_techniques[key]
        name = t["name"]
        name_trunc = name.replace(" ", "").replace("/","").replace("-","").replace("(", "").replace(")", "")
        mitre_id = t["external_id"]
        temp_list.append(mitre_id)
        technique_id = t["technique_id"]
        data_components = t["data_components"]
        kill_chain_phases = t["kill_chain_phases"]
        data_source_platforms = t["data_source_platforms"]
        mitigations = t["mitigations"]
        description = t["description"].replace("\n", "").replace('"', "'")
        atomic_tests = t["atomic_tests"]
        atomic_tests_str = "{"
        for at in atomic_tests:
            atomic_test_guid = at["auto_generated_guid"]
            atomic_test_init = f"AtomicTest({at})"
            atomic_tests_str += f"'{atomic_test_guid}': {atomic_test_init},"
        atomic_tests_str += "}"
        supported_os = list(
            set([os for at in atomic_tests for os in at["supported_platforms"]])
        )
        mapping[mitre_id] = name_trunc
        cwe_list = []
        cve_list = []
        mitre_to_cwe = {}
        cwe_to_cve = {}
        with open(metadata_path.joinpath('attack_to_cwe.json'), 'r') as f:
            mitre_to_cwe = json.load(f)
        with open(metadata_path.joinpath('cwe_to_cve.json'), 'r') as f:
            cwe_to_cve = json.load(f)

        mid = mitre_id.replace("T", "")
        pid = mitre_id.split(".")[0] if "." in mitre_id else mitre_id
        pid = pid.replace("T", "")

        if mid in list(mitre_to_cwe.keys()):
            cwe_list = mitre_to_cwe[mid]
            temp_cve_list = []
            for cwe in cwe_list:
                if cwe in list(cwe_to_cve.keys()):
                    temp_cve_list.extend(cwe_to_cve[cwe])
            if len(temp_cve_list) > 0:
                cve_list = list(set(temp_cve_list))
        elif pid in list(mitre_to_cwe.keys()):
            cwe_list = mitre_to_cwe[pid]
            temp_cve_list = []
            for cwe in cwe_list:
                if cwe in list(cwe_to_cve.keys()):
                    temp_cve_list.extend(cwe_to_cve[cwe])
            if len(temp_cve_list) > 0:
                cve_list = list(set(temp_cve_list))

        scripts += f"""
class {name_trunc}(Technique):
    mitre_id : str = "{mitre_id}"
    name : str = "{name}"
    technique_id : str = "{technique_id}"
    data_components : list[str] = {data_components}
    kill_chain_phases : list[str] = {kill_chain_phases}
    data_source_platforms : list[str] = {data_source_platforms}
    mitigations : list[str] = {mitigations}
    description : str = b"{"".join(c for c in description if ord(c)<128)}".decode('utf-8')
    atomic_tests : dict[str, AtomicTest] = {atomic_tests_str}
    supported_os : list[str] = {supported_os}
    cve_list : set[str] = {set(cve_list)}
    cwe_list : list[str] = {cwe_list}
"""
        preamble += f"'{mitre_id}': {name_trunc}, "
    preamble = preamble[:-2] + "}\n"
    scripts = scripts + preamble
    with open(metadata_path.joinpath('temp_techniques.py'), 'w') as f:
        f.write(scripts)

if __name__ == "__main__":
    generate_art_techniques()
