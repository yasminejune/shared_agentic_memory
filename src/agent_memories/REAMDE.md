  ## Folder architecutre

  # 
  ```bash
  config/ # What the system needs to know
  ```
  - Environment variables, constants, settings
  - Read at startup, rarely changes at runtime
  - Examples: settings.py, database_url, model_name, api_keys
  
  ```bash
  services/ # What the system does
  ```
  - Business logic and domain operations
  - Coordinates between data sources, external APIs, and your core logic
  - Stateful, often side-effectful (calls APIs, writes to DB)
  - Examples: LLMService, AuthService, TaskRunnerService

  ```bash
  utils/ #How the system does it
  ```
  - Pure, reusable helper functions with no business knowledge
  - Stateless, no side effects, easily testable in isolation
  - Could be copy-pasted into any project without modification
  - Examples: format_timestamp(), chunk_list(), retry_with_backoff()