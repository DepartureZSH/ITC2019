import pathlib
import torch
import numpy as np
import xml.etree.ElementTree as ET

class PSTTReader:
    def __init__(self, xml_path, matrix=False):
        self.path = pathlib.Path(xml_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        
        self.tree = ET.parse(str(xml_path))
        self.root = self.tree.getroot()

        # 根：problem/solution 二择一或二者并存（某些文件仅 problem，某些仅 solution，也可能 problem 内附 sample solution）
        if self.root.tag not in ("problem", "solution"):
            raise ValueError(f"Unsupported root tag: {self.root.tag}")
        
        # print(f"root : {self.root}")
        self.matrix = matrix

        # 公共元信息
        self.problem_name = None
        self.nrDays = None
        self.nrWeeks = None
        self.slotsPerDay = None
        
        self.timeTable_matrix = None


        # 各模块数据
        self.optimization = None
        self.rooms = {}
        self.rid_to_idx = {}
        self.travel = None
        self.courses = {}
        self.classes = {}
        self.cid_to_idx = {}
        self.students = {}
        self.sid_to_idx = {}
        self.distributions = []
        self.solution = None

        self._parse()

    # ---------- 顶层调度 ----------
    def _parse(self):
        self._parse_problem(self.root)
    
    # ---------- Problem ----------
    def _parse_problem(self, problem: ET.Element):
        # 根属性：name / nrDays / nrWeeks / slotsPerDay
        self.problem_name = problem.attrib.get("name")
        self.nrDays = self._to_int(problem.attrib.get("nrDays"))
        self.nrWeeks = self._to_int(problem.attrib.get("nrWeeks"))
        self.slotsPerDay = self._to_int(problem.attrib.get("slotsPerDay"))

        self.timeTable_matrix = np.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
        print(f"Problem Name: {self.problem_name}, Days: {self.nrDays}, Weeks: {self.nrWeeks}, Slots/Day: {self.slotsPerDay}")
        print(f"Initialized TimeTable matrix with shape: {self.timeTable_matrix.shape}")
        
        # optimization
        opt = problem.find("optimization")
        if opt is not None:
            self.optimization = {
                "time": self._to_int(opt.attrib.get("time"), 0),
                "room": self._to_int(opt.attrib.get("room"), 0),
                "distribution": self._to_int(opt.attrib.get("distribution"), 0),
                "student": self._to_int(opt.attrib.get("student"), 0),
            }

        # rooms
        rooms_node = problem.find("rooms")
        if rooms_node is not None:
            self.rooms, self.travel, self.rid_to_idx = self._parse_rooms(rooms_node)

        # courses
        courses_node = problem.find("courses")
        if courses_node is not None:
            self.courses, self.classes, self.cid_to_idx = self._parse_courses(courses_node)

        # distributions
        dist_node = problem.find("distributions")
        if dist_node is not None:
            self.distributions = self._parse_distributions(dist_node)

        # students
        students_node = problem.find("students")
        if students_node is not None:
            self.students, self.sid_to_idx = self._parse_students(students_node)

    # ---------- Rooms ----------
    def _parse_rooms(self, rooms_node):
        room_len = self._to_int(rooms_node.findall('room')[-1].attrib['id'])
        # print(f"rooms_node : {len(rooms_node.findall('room'))}")
        # print(f"rooms_node -1 id: {room_len}")
        result = {}
        travel = {}
        rid_to_idx = {}
        for i, r in enumerate(rooms_node.findall("room")):
            rid = self._to_int(r.attrib["id"])
            rid_to_idx[rid] = i
            cap = self._to_int(r.attrib.get("capacity"), 0)
            unavailables = []
            unavailable_zip = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
            unavailables_bits = []

            # travel
            for t in r.findall("travel"):
                other = t.attrib["room"]
                value = self._to_int(t.attrib.get("value"), 0)
                if not travel.get(r.attrib["id"], 0): travel[r.attrib["id"]] = {}
                travel[r.attrib["id"]].update({other: value})
                if not travel.get(other, 0): travel[other] = {}
                travel[other].update({r.attrib["id"]: value})

            # unavailable
            for u in r.findall("unavailable"):
                if self.matrix:
                    unavailable = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
                weeks_bits = u.attrib.get("weeks")
                if weeks_bits is not None:
                    weeks_list = self.bits_to_list(weeks_bits)
                days_bits = u.attrib.get("days")
                if days_bits is not None:
                    days_list = self.bits_to_list(days_bits)
                start = self._to_int(u.attrib.get("start"))
                length = self._to_int(u.attrib.get("length"))
                if start is not None and length is not None:
                    w_idx = torch.tensor(weeks_list, dtype=torch.long)
                    d_idx = torch.tensor(days_list, dtype=torch.long)
                    t_idx = torch.arange(start, start + length, dtype=torch.long)
                    if self.matrix:
                        W, D, T = torch.meshgrid(w_idx, d_idx, t_idx, indexing='ij')
                        unavailable[W, D, T] = 1
                        # unavailable[weeks_list, days_list, start: start + length] = 1
                if self.matrix:
                    unavailable_zip = torch.logical_or(unavailable_zip, unavailable)
                    unavailables.append(unavailable)
                    unavailables_bits.append((weeks_bits, days_bits, start, length))
            if self.matrix:
                room = {
                    "id": rid, 
                    "capacity": cap,
                    "unavailables_bits": unavailables_bits,
                    "unavailables": unavailables,
                    "unavailable_zip": unavailable_zip,
                    "ocupied": [] # (cid, time_bits, value)
                }
            else:
                room = {
                    "id": rid, 
                    "capacity": cap,
                    "unavailables_bits": unavailables_bits,
                    "ocupied": [] # (cid, time_bits, value)
                }
            result[r.attrib["id"]] = room
        return result, travel, rid_to_idx

    # # ---------- Courses / Config / Subpart / Class ----------
    def _parse_courses(self, courses_node):
        result = {}
        classes = {}
        cid_to_idx = {}
        for i, c in enumerate(courses_node.findall("course")):
            cid = self._to_int(c.attrib["id"])
            cid_to_idx[c.attrib["id"]] = i
            course = {
                "id": cid,
                "configs": {}
            }

            for cfg in c.findall("config"):
                cfg_id = cfg.attrib["id"]
                config = {
                    "id": cfg_id,
                    "subparts": {}
                }

                for sp in cfg.findall("subpart"):
                    sp_id = sp.attrib["id"]
                    subpart = {
                        "id": sp_id,
                        "classes": {}
                    }

                    for cl in sp.findall("class"):
                        cl_id = cl.attrib["id"]
                        limit = self._to_int(cl.attrib.get("limit")) if "limit" in cl.attrib else None
                        parent = cl.attrib.get("parent")
                        room_required = True
                        if "room" in cl.attrib and cl.attrib["room"].lower() == "false":
                            room_required = False

                        cdef = {
                            "id": cl_id,
                            "limit": limit,
                            "parent": parent,
                            "room_required": room_required,
                            "room_options": [],
                            "time_options": []
                        }

                        # 可选房间（含 penalty）
                        for rnode in cl.findall("room"):
                            cdef["room_options"].append({
                                "id":rnode.attrib["id"],
                                "penalty":self._to_int(rnode.attrib.get("penalty"), 0)
                            })


                        # 可选时间（含 penalty）
                        for tnode in cl.findall("time"):
                            if self.matrix:
                                optional_time = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
                            weeks_bits = tnode.attrib.get("weeks")
                            if weeks_bits is not None:
                                weeks_list = self.bits_to_list(weeks_bits)
                            days_bits = tnode.attrib.get("days")
                            if days_bits is not None:
                                days_list = self.bits_to_list(days_bits)
                            start = self._to_int(tnode.attrib.get("start"))
                            length = self._to_int(tnode.attrib.get("length"))
                            if start is not None and length is not None:
                                w_idx = torch.tensor(weeks_list, dtype=torch.long)
                                d_idx = torch.tensor(days_list, dtype=torch.long)
                                t_idx = torch.arange(start, start + length, dtype=torch.long)
                                if self.matrix:
                                    W, D, T = torch.meshgrid(w_idx, d_idx, t_idx, indexing='ij')
                                    optional_time[W, D, T] = 1
                                    # print("optional_time shape: ", optional_time.shape)
                                    # print(f"weeks_list : {weeks_list}, days_list: {days_list}, start: start + length: {start}: {start + length}")
                                    # optional_time[weeks_list, days_list, start: start + length] = 1
                            if self.matrix:
                                cdef["time_options"].append({
                                    "optional_time_bits": (weeks_bits, days_bits, start, length),
                                    "optional_time":optional_time,
                                    "penalty":self._to_int(tnode.attrib.get("penalty"), 0)
                                })
                            else:
                                cdef["time_options"].append({
                                    "optional_time_bits": (weeks_bits, days_bits, start, length),
                                    "penalty":self._to_int(tnode.attrib.get("penalty"), 0)
                                })
                        # Sort time_options by penalty
                        cdef["time_options"].sort(key=lambda x: x["penalty"])
                        subpart["classes"][cl_id] = cdef
                        classes[cl_id] = cdef
                    config["subparts"][sp_id] = subpart
                course["configs"][cfg_id] = config

            result[cid] = course
        return result, classes, cid_to_idx

    # # ---------- Distributions ----------
    def _parse_distributions(self, dist_node):
        results = []
        hard_constraints = []
        soft_constraints = []
        for d in dist_node.findall("distribution"):
            dtype = d.attrib["type"]
            required = d.attrib.get("required", "false").lower() == "true"
            penalty = self._to_int(d.attrib.get("penalty")) if "penalty" in d.attrib else None
            classes = [c.attrib["id"] for c in d.findall("class") if "id" in c.attrib]
            if required:
                hard_constraints.append({
                    "type": dtype, 
                    "required": required, 
                    "penalty": penalty,
                    "classes": classes
                })
            else:
                # self.soft_constraints.append(classes)
                soft_constraints.append({
                    "type": dtype, 
                    "required": required, 
                    "penalty": penalty,
                    "classes": classes
                })
        return {
            "hard_constraints": hard_constraints, 
            "soft_constraints": soft_constraints
        }

    # # ---------- Students ----------
    def _parse_students(self, students_node):
        results = {}
        sid_to_idx = {}
        for i, s in enumerate(students_node.findall("student")):
            sid = self._to_int(s.attrib["id"])
            sid_to_idx[sid] = i
            courses = [c.attrib["id"] for c in s.findall("course") if "id" in c.attrib]
            results[sid] = {
                "id": sid, 
                "courses": courses
            }
        return results, sid_to_idx

    # ---------- Solution（可作为根，或 problem 子节点） ----------
    def _parse_solution(self, node):
        meta = {
            "name": node.attrib.get("name"),
            "runtime": self._to_float(node.attrib.get("runtime")),
            "cores": self._to_int(node.attrib.get("cores")) if "cores" in node.attrib else None,
            "technique": node.attrib.get("technique"),
            "author": node.attrib.get("author"),
            "institution": node.attrib.get("institution"),
            "country": node.attrib.get("country"),
        }
        classes = {}
        for c in node.findall("class"):
            cid = c.attrib["id"]
            weeks_bits = c.attrib.get("weeks", "")
            if weeks_bits is not None:
                weeks_list = self.bits_to_list(weeks_bits)
            days_bits = c.attrib.get("days", "")
            if days_bits is not None:
                days_list = self.bits_to_list(days_bits)
            sc = {
                "id": cid,
                "days_bits": days_list,
                "start": self._to_int(c.attrib.get("start"), 0),
                "weeks_bits": weeks_list,
                "room": c.attrib.get("room"),
                "students": [s.attrib["id"] for s in c.findall("student") if "id" in s.attrib],
            }
            classes[cid] = sc
        print({
            "meta": meta, 
            "classes": classes
        })
        return {
            "meta": meta, 
            "classes": classes
        }

    # ---------- 工具 ----------
    @staticmethod
    def _to_int(x, default = None):
        if x is None:
            return default
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _to_float(x, default = None):
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default
        
    @staticmethod
    def bits_to_list(bits):
        return [i for i, bit in enumerate(list(bits)) if bit == "1"]

    def checkid(self):
        # if len(self.rid_to_idx.keys()) > 0:
            # print("room", self.rid_to_idx)
            # print(f"room length \t(last room id {list(self.rooms.keys())[-1]}):", len(self.rooms))
        if len(self.cid_to_idx.keys()) > 0: 
            print("courses", self.cid_to_idx)
            # print(f"courses length \t(last courses id {list(self.courses.keys())[-1]}): ", len(self.courses))
        # if len(self.sid_to_idx.keys()) > 0: 
            # print("students", self.sid_to_idx)
            # print(f"students length \t(last student id {list(self.students.keys())[-1]}):", len(self.students))

    def describe(self):
        # 各模块数据
        print(self.optimization)
        print(f"room length (last room id {list(self.rooms.keys())[-1]}):", len(self.rooms))
        first_room = self.rooms[list(self.rooms.keys())[0]]
        if len(first_room["unavailables"]) > 0:
            first_unavailable_bits = first_room["unavailables_bits"][0]
            first_unavailable_shape = first_room["unavailables"][0].shape
            print(first_unavailable_bits[0])
            # print(optional_time_to_bits(first_room["unavailables"][0]))
        else:
            first_unavailable_bits = None
            first_unavailable_shape = None
        print("    A room: id={} capacity={} unavailables_bits={} unavailables shape={}".format(first_room["id"], first_room["capacity"], first_unavailable_bits, (len(first_room["unavailables"]), first_unavailable_shape)))
        print("    travel matrix: shape={} all zeros={}".format(self.travel.shape, torch.all(self.travel == 0)))
        print(f"courses length (last courses id {list(self.courses.keys())[-1]}): ", len(self.courses))
        first_courses = self.courses[list(self.courses.keys())[0]]
        print("    A course: id={} configs length={} (last config id {})".format(first_courses["id"], len(first_courses["configs"]), list(first_courses["configs"].keys())[-1]))
        _first_config = first_courses["configs"][list(first_courses["configs"].keys())[0]]
        print("        A config: id={} subparts length={} (last subpart id {})".format(_first_config["id"], len(_first_config["subparts"]), list(_first_config["subparts"].keys())[-1]))
        _first_subpart = _first_config["subparts"][list(_first_config["subparts"].keys())[0]]
        print("            A subpart: id={} classes length={} (last class id {})".format(_first_subpart["id"], len(_first_subpart["classes"]), list(_first_subpart["classes"].keys())[-1]))
        _first_class = _first_subpart["classes"][list(_first_subpart["classes"].keys())[0]]
        # print(_first_class.keys())
        if len(_first_class["room_options"]) > 0:
            _room_options = (len(_first_class["room_options"]), _first_class["room_options"][0])
        else:
            _room_options = None
        if len(_first_class["time_options"]) > 0:
            _time_options = (len(_first_class["time_options"]), "optional_time_bits: {}".format(_first_class["time_options"][0]["optional_time_bits"]),"optional_time: {}".format(_first_class["time_options"][0]["optional_time"].shape), "penalty: {}".format(_first_class["time_options"][0]["penalty"]))
        else:
            _time_options = None

        print("                A class: id={} limit={} parent={} room_required={} room_options={} time_options={}".format(_first_class["id"], _first_class["limit"], _first_class["parent"], _first_class["room_required"], _room_options, _time_options))
        first_students = self.students[list(self.students.keys())[0]]
        # print(first_students.keys())
        print(f"students length (last student id {list(self.students.keys())[-1]}):", len(self.students))
        print("    A students id={} courses length={}".format(first_students["id"], len(first_students["courses"])))
        print("        A student course {}".format(first_students["courses"]))

        print(f"hard distributions length :", len(self.distributions["hard_constraints"]))
        print("    A hard distribution: type={} required={} penalty={} classes length={}".format(self.distributions["hard_constraints"][0]["type"], self.distributions["hard_constraints"][0]["required"], self.distributions["hard_constraints"][0]["penalty"], len(self.distributions["hard_constraints"][0]["classes"])))
        print("        A hard distribution class {}".format(self.distributions["hard_constraints"][0]["classes"]))

        print("soft distributions length:", len(self.distributions["soft_constraints"]))
        print("    A soft distribution: type={} required={} penalty={} classes length={}".format(self.distributions["soft_constraints"][0]["type"], self.distributions["soft_constraints"][0]["required"], self.distributions["soft_constraints"][0]["penalty"], len(self.distributions["soft_constraints"][0]["classes"])))
        print("        A soft distribution class {}".format(self.distributions["soft_constraints"][0]["classes"]))

    def describe_PSTT(self):
        print("==================================================================================")
        print(f"This project is to allocate {len(self.rooms)} rooms and schedule a timetable for {len(self.classes)} classes.")
        print(f"Solution should satisfy all {len(self.distributions['hard_constraints'])} hard_constraints and try to minimize penalty.")
        # print(f"One class example key={list(self.classes.keys())[0]}:")
        # print(self.classes[list(self.classes.keys())[1]])
        # print(f"One room example key={type(list(self.rooms.keys())[0])}:")
        # print(self.rooms[list(self.rooms.keys())[0]])
        for each in self.distributions['hard_constraints']:
            # if each['type'] == "Precedence":
            #     for cid in each['classes']:
            #         print("cid ", cid, " ", each)
            #         # ind = self.cid_to_idx[cid]
            #         print(self.classes[cid]["time_options"][0]["optional_time_bits"])
            #     exit(1)
            if each['type'].startswith("MaxDayLoad"):
                for cid in each['classes']:
                    print("cid ", cid, " ", each)
                    # ind = self.cid_to_idx[cid]
                    print(self.classes[cid]["time_options"][0]["optional_time_bits"])
                exit(1)
        for each in self.distributions['soft_constraints']:
            # if each['type'] == "Precedence":
            #     for cid in each['classes']:
            #         print("cid ", cid, " ", each)
            #         # ind = self.cid_to_idx[cid]
            #         print(self.classes[cid]["time_options"][0]["optional_time_bits"])
            #     exit(1)
            if each['type'].startswith("MaxDayLoad"):
                for cid in each['classes']:
                    print("cid ", cid, " ", each)
                    # ind = self.cid_to_idx[cid]
                    print(self.classes[cid]["time_options"][0]["optional_time_bits"])
                exit(1)
        # print(self.distributions['hard_constraints'])

# 测试函数
# def test_optional_time_to_bits():
#     # 创建一个测试的 optional_time 张量
#     nrWeeks, nrDays, slotsPerDay = 16, 7, 288
#     optional_time = torch.zeros((nrWeeks, nrDays, slotsPerDay), dtype=torch.float32)

#     # 设置一些时间槽
#     weeks_bits = "1111111111111111"
#     days_bits = "1010000"
#     start = 96
#     length = 15

#     # 将时间槽设置为 1
#     weeks_list = [i for i, bit in enumerate(weeks_bits) if bit == "1"]
#     days_list = [i for i, bit in enumerate(days_bits) if bit == "1"]
#     for w in weeks_list:
#         for d in days_list:
#             optional_time[w, d, start:start + length] = 1

#     # 调用函数进行转换
#     result = optional_time_to_bits(optional_time)

#     # 打印结果
#     print("Expected:", (weeks_bits, days_bits, start, length))
#     print("Result:", result)

#     # 验证结果是否正确
#     assert result == (weeks_bits, days_bits, start, length), "Test failed!"
#     print("Test passed!")

if __name__ == "__main__":
    import os
    import yaml

    folder = pathlib.Path(__file__).parent.resolve()
    with open(f"{folder}/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    # 读取单个 XML（既可是 problem.xml，也可直接是 solution.xml）
    # file = "/home/unnc/ZSH/Projects/itc2019/data/late/bet-spr18.xml"
    file = f'{config["data"]["folder"]}/{config["data"]["file"]}'
    reader = PSTTReader(file, matrix=False)
    reader.describe_PSTT()
    class1 = reader.classes['3']
    print("room_options:", len(class1['room_options']))
    print("time_options:", len(class1['time_options']))
    # print(reader.checkid())


    # folder = "/home/unnc/ZSH/Projects/itc2019/data/late/"
    # folder = "/home/unnc/ZSH/Projects/itc2019/solutions/"
    # for each in os.listdir(folder):
    #     if each.endswith(".xml"):
    #         file = f"{folder}/{each}"
    #         print(f"Reading file: {file}")
            
    #         reader = UCTPReader(file)
    #         reader.checkid()