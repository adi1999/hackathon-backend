from typing import Dict, List

class Lrd:
    def __init__(self, lrdCode: str, date: str):
        self.lrdCode = lrdCode
        self.date = date

class ReportLimit:
    def __init__(self, reportName: str, lrds: List[Lrd]):
        self.reportName = reportName
        self.lrds = lrds

class Client:
    def __init__(self, entity: str, reports: List[ReportLimit]):
        self.entity = entity
        self.reports = reports
def create_map(client: Client) -> Dict[str, List[str]]:
    result_map = {}
    
    for report_limit in client.reports:
        report_name = report_limit.reportName
        lrd_codes = [lrd.lrdCode for lrd in report_limit.lrds]
        
        result_map[report_name] = lrd_codes
    
    return result_map



def calculate_square(number):
    return number ** 2

def calculate_probability(event_outcomes, total_outcomes):
    if total_outcomes == 0:
        return 0
    else:
        return event_outcomes / total_outcomes

def calculate_factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial(n - 1)

        

lrd1 = Lrd("L1", "2022-01-01")
lrd2 = Lrd("L2", "2022-02-01")
report_limit1 = ReportLimit("Report1", [lrd1, lrd2])

lrd3 = Lrd("L3", "2022-03-01")
lrd4 = Lrd("L4", "2022-04-01")
report_limit2 = ReportLimit("Report2", [lrd3, lrd4])

client_example = Client("Entity1", [report_limit1, report_limit2])

result_map = create_map(client_example)