
import sys
import os

project_root = "/Users/ashwin.s/Desktop/CFO-RL/Multi-Agent-RL-based-Cashflow-Manager"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Current Working Directory: {os.getcwd()}")
print(f"Project Root: {project_root}")

try:
    from models import CashflowmanagerAction
    print("✅ SUCCESS: Import from 'models' successful")
except ImportError as e:
    print(f"❌ FAILURE: Import from 'models' failed: {e}")

try:
    from server.cashflowmanager_environment import CashflowmanagerEnvironment
    print("✅ SUCCESS: Import from 'server.cashflowmanager_environment' successful")
except ImportError as e:
    print(f"❌ FAILURE: Import from 'server.cashflowmanager_environment' failed: {e}")

try:
    from server.client import get_model_response
    print("✅ SUCCESS: Import from 'server.client' successful")
except ImportError as e:
    print(f"❌ FAILURE: Import from 'server.client' failed: {e}")

try:
    from server.agents import expenditure_agent
    print("✅ SUCCESS: Import from 'server.agents' successful")
except ImportError as e:
    print(f"❌ FAILURE: Import from 'server.agents' failed: {e}")

try:
    from openenv.core.env_server.interfaces import Environment
    print("✅ SUCCESS: Import 'openenv.core.env_server.interfaces' successful")
except ImportError as e:
    print(f"❌ FAILURE: Import 'openenv.core.env_server.interfaces' failed: {e}")
