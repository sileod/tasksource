"""DiscoGeM's native PDTB3 codes, in the order used by the mirror's vote vectors."""
DISCO_RELATIONS = {
    'synchronous':'events overlap in time', 'precedence':'text_A happens before text_B',
    'succession':'text_B happens before text_A', 'reason':'text_B explains the cause of text_A',
    'result':'text_B is a result of text_A', 'arg1-as-cond':'text_A is a condition for text_B',
    'arg2-as-cond':'text_B is a condition for text_A', 'arg1-as-negcond':'text_A is a negative condition for text_B',
    'arg2-as-negcond':'text_B is a negative condition for text_A', 'arg1-as-goal':'text_A is the goal of text_B',
    'arg2-as-goal':'text_B is the goal of text_A', 'arg1-as-denier':'text_A denies an expectation from text_B',
    'arg2-as-denier':'text_B denies an expectation from text_A', 'contrast':'contrast', 'similarity':'similarity',
    'conjunction':'additional information', 'disjunction':'alternatives', 'arg1-as-instance':'text_A exemplifies text_B',
    'arg2-as-instance':'text_B exemplifies text_A', 'arg1-as-detail':'text_A elaborates text_B',
    'arg2-as-detail':'text_B elaborates text_A', 'equivalence':'equivalent descriptions',
    'arg1-as-manner':'text_A describes how text_B happens', 'arg2-as-manner':'text_B describes how text_A happens',
    'arg1-as-excpt':'text_A is an exception to text_B', 'arg2-as-excpt':'text_B is an exception to text_A',
    'arg2-as-subst':'text_B substitutes for text_A', 'differentcon':'another connective relation', 'norel':'no discourse relation',
}
