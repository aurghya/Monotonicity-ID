from collections.abc import Iterable

class CTFTerm:

    def __init__(self, vars: dict, do_var:dict):
        self.vars = vars
        self.do_var = do_var

    def __str__(self):
        # TODO: Make it more readable
        return f"CTFTerm({self.vars}, {self.do_var})"
    
    def __eq__(self, other):
        # Check if the two terms are equal
        return self.vars == other.vars and self.do_var == other.do_var
    
    def __hash__(self):
        # Hash the term
        return hash(frozenset(self.vars.items()), frozenset(self.do_var.items()))


class CTF:

    def __init__(self, term_set: set, cond_term_set: set = None):

        # If the terms are are not empty they are added to the set
        self.term_set = {term for term in term_set if term.vars}
        self.cond_term_set = {term for term in cond_term_set if term.vars} if cond_term_set else set()

    def add_term(self, term: CTFTerm):
        # Add a term to the set
        if term.vars:
            self.term_set.add(term)
    
    def add_cond_term(self, term: CTFTerm):
        # Add a conditional term to the set
        if term.vars:
            self.cond_term_set.add(term)

    def __str__(self):
        # TODO: Make it more readable
        formatted_terms = ", ".join([str(term) for term in self.term_set])
        return f"CTF({formatted_terms})"
    
def split_ctf(q: CTF):
    # Split the CTF into individual terms
    new_terms = set()
    new_cond_terms = set()

    for term in q.term_set:
        for var, val in term.items():
            new_terms.add(CTFTerm({var: val}, term.do_var))
    
    for term in q.cond_term_set:
        for var, val in term.items():
            new_cond_terms.add(CTFTerm({var: val}, term.do_var))

    return CTF(new_terms, new_cond_terms)