import sys
from pathlib import Path

# scripts/ holds standalone entry points (not part of the installed package),
# but its pure logic still needs to be importable from tests.
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
