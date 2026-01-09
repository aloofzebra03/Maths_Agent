from pydantic import BaseModel

class ConceptPkg(BaseModel):
    title: str

# concept_pkg is now dynamically created from state["concept_title"]
# No longer using a global hardcoded value

