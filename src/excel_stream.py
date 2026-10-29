"""Stream the provided alarm OOXML without loading the full worksheet in RAM."""

import hashlib
import zipfile
from datetime import datetime

from lxml import etree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
EXPECTED = [
    "Driver",
    "Operation line",
    "Alarm type",
    "speed(km/h)",
    "Alarm start time",
    "Alarm end time",
    "Alarm duration",
    "StartDate",
    "MonthYear",
    "WeekYear",
]


def normalize_key(value):
    if value is None or not str(value).strip():
        return None
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip().upper()


def parse_date(value):
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        try:
            return datetime.strptime(str(value).strip(), "%d-%b-%Y (%a)")
        except ValueError:
            return None


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def alarm_rows(path):
    with zipfile.ZipFile(path) as book:
        strings = []
        with book.open("xl/sharedStrings.xml") as stream:
            for _, element in ET.iterparse(stream, events=("end",), tag=NS + "si"):
                strings.append("".join(element.itertext()))
                element.clear()
                while element.getprevious() is not None:
                    del element.getparent()[0]
        relationships = ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in relationships}
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        if workbook.find(NS + "workbookPr").get("date1904", "0") not in ("0", "false"):
            raise ValueError("The source uses an unsupported 1904 date system")
        for sheet in workbook.find(NS + "sheets"):
            name = sheet.get("name")
            target = targets[sheet.get(REL + "id")]
            target = target.lstrip("/") if target.startswith("/") else "xl/" + target
            with book.open(target) as stream:
                for _, row in ET.iterparse(stream, events=("end",), tag=NS + "row"):
                    values = [None] * 10
                    for cell in row:
                        letters = "".join(c for c in cell.get("r", "") if c.isalpha())
                        column = 0
                        for letter in letters:
                            column = column * 26 + ord(letter) - 64
                        if not 1 <= column <= 10:
                            continue
                        value = cell.findtext(NS + "v")
                        if cell.get("t") == "s" and value is not None:
                            value = strings[int(value)]
                        elif cell.get("t") == "inlineStr":
                            value = "".join(cell.find(NS + "is").itertext())
                        values[column - 1] = value
                    yield name, int(row.get("r")), values
                    row.clear()
                    while row.getprevious() is not None:
                        del row.getparent()[0]
