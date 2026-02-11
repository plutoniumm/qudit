from traceback import format_exception as fmx
import unittest
import torch
import time
import os

outDir = "../docs/tests"

load = unittest.defaultTestLoader.loadTestsFromTestCase
C64 = torch.complex64


class Result(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._pending = {}

    def addSuccess(self, test):
        self._pending[test] = ("PASS", None)

    def addFailure(self, test, err):
        self._pending[test] = ("FAIL", err)

    def addError(self, test, err):
        self._pending[test] = ("ERROR", err)

    def startTest(self, test):
        super().startTest(test)
        setattr(test, "_mdr_start_time", time.perf_counter())

    def stopTest(self, test):
        start = getattr(test, "_mdr_start_time", None)
        setattr(
            test,
            "_mdr_elapsed",
            (time.perf_counter() - start) if start is not None else None,
        )

        status, err = self._pending.pop(test, ("PASS", None))
        self.rows.append(self._row(test, status, err=err))
        super().stopTest(test)

    def _row(self, test, status, err=None):
        elapsed = getattr(test, "_mdr_elapsed", None) or 0.0

        if status == "PASS":
            msg = "✓"
        else:
            msg = "Test failed"
            if err is not None:
                formatted = fmx(*err)
                formatted = [line.strip() for line in formatted if line.strip()]

                if formatted:
                    msg = formatted[-1]

        return {
            "name": test._testMethodName,
            "desc": test.shortDescription() or "",
            "result": msg,
            "time": float(elapsed),
        }


class Exam:
    def __init__(self, name, desc, file):
        self.name = name
        self.desc = desc

        if not os.path.exists(outDir):
            os.makedirs(outDir)
        self.file = os.path.join(outDir, file)

        with open(self.file, "w") as f:
            f.write(f"# {self.name}\n")
            f.write(f"{self.desc}\n\n")

    def run(self, test):
        suite_name = "Unknown Suite"
        suite_desc = ""

        head = test
        while isinstance(head, unittest.TestSuite):
            head = next(iter(head))

        suite_name = head.__class__.__name__
        suite_desc = head.__class__.__doc__

        result = Result()
        test(result)
        self.write(result, suite_name, suite_desc)

        return result

    def write(self, result, name, desc):
        with open(self.file, "a") as f:
            f.write(f"\n## {name}\n")
            if desc:
                f.write(f"{desc.strip()}\n\n")

            f.write("| Test name | Description | Result | Time taken (s) |\n")
            f.write("|----------|-------------|--------|----------------|\n")
            for r in result.rows:
                tname = r["name"].replace("test_", "")
                taken = f"{r.get('time', 0.0):.4f}"
                res = str(r.get("result", ""))
                desc = str(r.get("desc", "")).replace("\n", " ").replace("|", "\|")
                res = res.replace("\n", " ")

                f.write(f"| {tname} | {desc} | {res} | {taken} |\n")


class Question(unittest.TestCase):

    def tensorise(self, x):
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            x = torch.tensor(x)

        return torch.tensor(x, dtype=C64)

    def stateEqual(self, a, b, atol=1e-7, msg="Unmatched result"):
        ta = torch.as_tensor(a, dtype=C64).reshape(-1)
        tb = torch.as_tensor(b, dtype=C64).reshape(-1)
        if ta.shape != tb.shape:
            return False

        na = torch.linalg.vector_norm(ta)
        nb = torch.linalg.vector_norm(tb)
        if na == 0 or nb == 0:
            return torch.allclose(ta, tb, atol=atol)

        ta = ta / na
        tb = tb / nb

        k = int(torch.argmax(torch.abs(tb)).item())

        if torch.abs(tb[k]).item() < atol:
            return torch.allclose(ta, tb, atol=atol)

        phase = ta[k] / tb[k]
        eq = torch.allclose(ta, phase * tb, atol=atol)
        self.assertEqual(eq, True, msg=msg)

    def matEqual(self, A, B, places=4):
        A = self.tensorise(A)
        B = self.tensorise(B)

        self.assertEqual(tuple(A.shape), tuple(B.shape))

        diff = round(torch.sum(torch.abs(A - B)).item(), int(places))
        self.assertEqual(diff, 0)
