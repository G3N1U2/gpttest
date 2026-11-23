import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

ORCHESTRATORS = {
    "pipeline.py": ["pipeline"],
    "stages.py": [
        "sample",
        "split",
        "tokenize",
        "embed",
        "build_gpt",
        "predict",
        "train",
        "test",
    ],
}


def functions_of(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def body_without_docstring(function):
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    return body


class CallsOnlyTest(unittest.TestCase):
    def test_orchestrators_exist(self):
        for filename, names in ORCHESTRATORS.items():
            functions = functions_of(ROOT / filename)
            for name in names:
                with self.subTest(function=name):
                    self.assertIn(name, functions, f"{filename}: нет функции {name}()")

    def test_orchestrators_are_calls_only(self):
        for filename, names in ORCHESTRATORS.items():
            functions = functions_of(ROOT / filename)
            for name in names:
                with self.subTest(function=name):
                    self.check_function(functions[name], filename)

    def check_function(self, function, filename):
        where = f"{filename}:{function.name}"
        body = body_without_docstring(function)
        self.assertTrue(body, f"{where}: пустое тело")

        for statement in body:
            if isinstance(
                statement,
                (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.AugAssign, ast.Raise, ast.Delete),
            ):
                self.fail(
                    f"{where}: строка {statement.lineno} — {type(statement).__name__}. "
                    "Функция-этап должна состоять только из вызовов."
                )
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                self.assertIsInstance(
                    statement.value,
                    ast.Call,
                    f"{where}: строка {statement.lineno} — не вызов",
                )
            elif isinstance(statement, ast.Return):
                self.check_return(statement, where)
            else:
                self.fail(
                    f"{where}: строка {statement.lineno} — недопустимая инструкция "
                    f"{type(statement).__name__}"
                )

    def check_return(self, statement, where):
        value = statement.value
        if value is None or isinstance(value, (ast.Name, ast.Call)):
            return
        self.fail(
            f"{where}: строка {statement.lineno} — return должен возвращать вызов "
            "или переменную"
        )


if __name__ == "__main__":
    unittest.main()
