"""
Functions for reorganizing a file's imports nicely
"""
import distutils.sysconfig as sysconfig
import os
import platform


_d_map = {
    "Linux": "/d",
    "Windows": "D:\\"
}
d_drive = _d_map[platform.system()]


def std_modules():
    # finding modules in the standard library
    # https://stackoverflow.com/questions/46441539/how-to-determine-if-a-module-name-is-part-of-python-standard-library?rq=1
    ret_list = []
    std_lib = sysconfig.get_python_lib(standard_lib=True)
    for top, dirs, files in os.walk(std_lib):
        for nm in files:
            if nm != '__init__.py' and nm[-3:] == '.py':
                ret_list.append(os.path.join(top, nm)[len(std_lib)+1:-3].
                                replace('\\','.'))
    return ret_list


def compare_modules(modules):
    modules = set(modules)
    std = set(std_modules())

    return std.intersection(modules)


def sort_imports(import_list):
    full_imports = []
    part_imports = []
    part_imports_dict = dict()
    # get rid of duplicated full import statements
    import_list = list(set(import_list))
    for import_statement in import_list:
        import_statement_list = import_statement.split()
        if import_statement_list[0] == 'import':
            full_imports.append(import_statement)
        else:
            # these are 'from' imports
            if 'as' in import_statement:
                part_imports.append(import_statement)
            else:
                module = import_statement_list[1]
                if module in part_imports_dict.keys():
                    for item in import_statement_list[3:]:
                        part_imports_dict[module].append(item)
                else:
                    part_imports_dict[module] = import_statement_list[3:]
    full_imports = sorted(full_imports, key=str.lower)
    for pkg, pkg_imports in part_imports_dict.items():
        pkg_imports = ', '.join(pkg_imports)
        part_imports.append(
            reorder_multi_import(
                'from ' + pkg + ' import ' + pkg_imports
            )
        )
    part_imports = sorted(part_imports, key=str.lower)
    return full_imports + part_imports


def reorder_multi_import(import_line):
    import_line = import_line.replace(',', '').strip().split()
    current_imports = import_line[3:]
    current_imports = sorted(current_imports, key=str.lower)
    return ' '.join(import_line[:3]) + ' ' + ', '.join(current_imports)


def fix_imports(file_path):
    stdlib = std_modules()
    stdlib_imports = []
    third_party_imports = []
    local_imports = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            l = line.strip().split()
            if 'import' in l:
                if ',' in line:
                    line = reorder_multi_import(line)
                else:
                    line = ' '.join(l)
                if l[0] in ('from', 'import'):
                    pkg = l[1]
                    if pkg[0] == '.':
                        local_imports.append(line)
                    elif pkg in stdlib:
                        stdlib_imports.append(line)
                    else:
                        third_party_imports.append(line)

    return (
               sort_imports(stdlib_imports),
               sort_imports(third_party_imports),
               sort_imports(local_imports)
    )


def print_fixed_imports(file_path):
    imp_standard, imp_third, imp_local = fix_imports(file_path)

    if len(imp_standard):
        print('# stdlib imports')
        for import_statement in imp_standard:
            print(import_statement)
        print('')
    if len(imp_third):
        print('# third party imports')
        for import_statement in imp_third:
            print(import_statement)
        print('')
    if len(imp_local):
        print('# local imports')
        for import_statement in imp_local:
            print(import_statement)


if __name__ == '__main__':
    test_path = r'lol.py'
    print_fixed_imports(test_path)
