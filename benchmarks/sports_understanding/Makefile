workflow:
	uv run expt.py run evaluate.expt_name=workflow \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.sports_understanding_workflow

pot:
	uv run expt.py run evaluate.expt_name=pot \
	  ptools.are_sports_in_sentence_consistent.method=program_of_thought \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

react:
	uv run expt.py run evaluate.expt_name=react \
	  ptools.are_sports_in_sentence_consistent.method=simulate_pydantic \
	  ptools.are_sports_in_sentence_consistent.tools=[ptools.analyze_sentence,ptools.sport_for,ptools.consistent_sports]

zeroshot_structured: 
	uv run expt.py run evaluate.expt_name=zeroshot_structured \
	  ptools.are_sports_in_sentence_consistent.method=simulate

zeroshot_unstructured: 
	uv run expt.py run evaluate.expt_name=zeroshot_unstructured \
	  ptools.are_sports_in_sentence_consistent.method=direct \
	  ptools.are_sports_in_sentence_consistent.fn=ptools.zeroshot_unstructured_workflow

list:
	uv run python -m secretagent.cli.results list results/*

avg:
	uv run python -m secretagent.cli.results average --metric correct --metric cost results/*

pair:
	uv run python -m secretagent.cli.results pair --metric correct --metric cost results/*

compare:
	uv run python -m secretagent.cli.results compare-configs results/*
