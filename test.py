import importlib.util

# Specify the path to the .pyc file
pyc_path = '__pycache__/SVMFeat1.cpython-39.pyc'

# Create a module spec from the .pyc file
spec = importlib.util.spec_from_file_location('SVMFeat1', pyc_path)

# Import the module
svf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svf)

# Now you can use the imported module
svf.SVMFeat1()
