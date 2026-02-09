"""
Helper utilities for benchmark tests.
"""
import os
import tempfile
import yaml


def get_config_path_with_agent_override(original_config: str, agent_type: str | None) -> str:
    """
    Returns the config path, optionally creating a temp file with modified agent type.
    
    Args:
        original_config: Path to the original YAML config (relative to configs dir)
        agent_type: If set, override the agent type in the config
        
    Returns:
        Path to the config file to use (original or temp modified)
    """
    # Resolve the original config path
    configs_dir = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "../../mcpuniverse/benchmark/configs"
    )
    config_path = os.path.join(configs_dir, original_config)
    
    if not agent_type:
        return original_config  # Return original relative path

    assert agent_type in ["react", "function_call"], "Invalid agent type"
    
    # Read and modify the config
    with open(config_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))
    
    # Modify agent type in agent definitions
    # Config format: kind: agent, spec: {name: ..., type: react/function_call}
    new_agent_name = agent_type + "-agent"
    for doc in docs:
        if doc.get("kind", "").lower() == "agent" and "spec" in doc:
            original_type = doc["spec"].get("type", "unknown")
            doc["spec"]["type"] = agent_type
            doc["spec"]["name"] = new_agent_name
            print(f"Overriding agent type from '{original_type}' to '{agent_type}', name to '{new_agent_name}'")
        elif doc.get("kind", "").lower() == "benchmark" and "spec" in doc:
            # Update the benchmark's agent reference to match the new agent name
            original_agent = doc["spec"].get("agent", "unknown")
            doc["spec"]["agent"] = new_agent_name
            print(f"Overriding benchmark agent reference from '{original_agent}' to '{new_agent_name}'")
    
    # Write to a temp file
    temp_file = tempfile.NamedTemporaryFile(
        mode="w", 
        suffix=".yaml", 
        delete=False,
        prefix="benchmark_config_"
    )
    yaml.dump_all(docs, temp_file, default_flow_style=False)
    temp_file.close()
    
    return temp_file.name


def cleanup_temp_config(config_path: str, agent_type: str | None) -> None:
    """Clean up temp config file if it was created."""
    if agent_type and config_path and os.path.exists(config_path) and config_path.startswith(tempfile.gettempdir()):
        os.unlink(config_path)
