from agentic_secretary.graph import PlannerState, build_graph


def test_graph_runs_start_to_end():
    graph = build_graph()

    result = graph.invoke(
        {"status": "pending"}, config={"configurable": {"thread_id": "test"}}
    )

    assert result["status"] == "done"


def test_planner_state_has_status_field():
    assert "status" in PlannerState.__annotations__
