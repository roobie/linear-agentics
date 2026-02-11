# Linear Agent Runtime: Design Document

## Vision
Build a minimal proof-of-concept agent runtime where resource safety is enforced through linear types. An LLM generates agent code in a simple DSL, the type system validates it, and compilation errors feed back to the LLM for self-repair. Focus on one concrete use case: analyzing GitHub issues and proposing fixes.

## Core Principle
**Resources are physical objects that must be used exactly once.** Rust's ownership system (no Clone/Copy) enforces linearity automatically. If agent code tries to reuse a token, it won't compile.

---

## Architecture Overview

```
User Request
    ↓
LLM generates DSL code
    ↓
DSL Parser → Rust codegen
    ↓
Rust compiler (type checking)
    ↓
[If errors] → Feed back to LLM → Retry
    ↓
[If success] → Execute with linear tokens
    ↓
Result + Audit Proofs
```

Three layers:
1. **Linear Runtime** (Rust): Provides token types and operations
2. **Agent DSL**: Simple language LLMs generate
3. **LLM Integration**: Generation + error-driven repair

---

## Part 1: Linear Runtime (Rust)

### Core Token Types

All tokens are **non-Clone, non-Copy** structs. Using them consumes them.

```rust
// src/tokens.rs

/// Computational budget - consumed by operations
pub struct Budget {
    remaining: u32,
}

impl Budget {
    pub fn new(amount: u32) -> Self {
        Budget { remaining: amount }
    }

    /// Spend budget, return new budget with reduced amount
    pub fn spend(self, amount: u32) -> Result<Budget, BudgetExhausted> {
        if self.remaining >= amount {
            Ok(Budget { remaining: self.remaining - amount })
        } else {
            Err(BudgetExhausted {
                requested: amount,
                available: self.remaining,
            })
        }
    }

    /// Split budget between parallel operations
    pub fn split(self) -> (Budget, Budget) {
        let half = self.remaining / 2;
        (Budget { remaining: half }, Budget { remaining: self.remaining - half })
    }

    pub fn remaining(&self) -> u32 {
        self.remaining
    }
}

/// File read access - scoped to specific path
pub struct FileReadToken {
    root_path: PathBuf,
}

impl FileReadToken {
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        FileReadToken { root_path: root_path.into() }
    }
}

/// GitHub read access - scoped to specific repo
pub struct GithubReadToken {
    repo: String,
    auth_token: String,
}

impl GithubReadToken {
    pub fn new(repo: impl Into<String>, auth_token: impl Into<String>) -> Self {
        GithubReadToken {
            repo: repo.into(),
            auth_token: auth_token.into(),
        }
    }
}

/// LLM API access - for calling Claude/GPT
pub struct LlmToken {
    api_key: String,
    model: String,
}

impl LlmToken {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        LlmToken {
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}
```

### Proof Types

Every operation returns a **proof of use** that must be consumed (typically by logging).

```rust
// src/proofs.rs

use std::time::SystemTime;

/// Proof that a file was read
#[derive(Debug)]
pub struct FileReadProof {
    path: PathBuf,
    timestamp: SystemTime,
    bytes_read: usize,
}

/// Proof that GitHub API was called
#[derive(Debug)]
pub struct GithubApiProof {
    endpoint: String,
    timestamp: SystemTime,
    status: u16,
}

/// Proof that LLM was called
#[derive(Debug)]
pub struct LlmCallProof {
    model: String,
    timestamp: SystemTime,
    tokens_used: u32,
}

/// Aggregate proof bundle
#[derive(Debug, Default)]
pub struct AuditTrail {
    file_reads: Vec<FileReadProof>,
    github_calls: Vec<GithubApiProof>,
    llm_calls: Vec<LlmCallProof>,
}

impl AuditTrail {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_file_read(&mut self, proof: FileReadProof) {
        self.file_reads.push(proof);
    }

    pub fn record_github_call(&mut self, proof: GithubApiProof) {
        self.github_calls.push(proof);
    }

    pub fn record_llm_call(&mut self, proof: LlmCallProof) {
        self.llm_calls.push(proof);
    }

    pub fn summary(&self) -> String {
        format!(
            "Files read: {}, GitHub calls: {}, LLM calls: {}",
            self.file_reads.len(),
            self.github_calls.len(),
            self.llm_calls.len()
        )
    }
}
```

### Operations

Operations consume tokens and return results + proofs.

```rust
// src/operations.rs

use anyhow::Result;

/// Read a file using the token
pub fn read_file(
    token: FileReadToken,
    path: &str,
    budget: Budget,
) -> Result<(String, FileReadProof, Budget)> {
    let budget = budget.spend(50)?; // Reading costs 50 units

    let full_path = token.root_path.join(path);

    // Security check: ensure path is within token scope
    if !full_path.starts_with(&token.root_path) {
        anyhow::bail!("Path traversal attempt: {}", path);
    }

    let contents = std::fs::read_to_string(&full_path)?;
    let proof = FileReadProof {
        path: full_path,
        timestamp: SystemTime::now(),
        bytes_read: contents.len(),
    };

    Ok((contents, proof, budget))
}

/// Fetch GitHub issue
pub fn fetch_github_issue(
    token: GithubReadToken,
    issue_number: u32,
    budget: Budget,
) -> Result<(GitHubIssue, GithubApiProof, Budget)> {
    let budget = budget.spend(100)?; // API call costs 100 units

    // Make actual GitHub API call
    let url = format!("https://api.github.com/repos/{}/issues/{}", token.repo, issue_number);

    // TODO: Real HTTP request with token.auth_token
    // For now, mock response
    let issue = GitHubIssue {
        number: issue_number,
        title: "Mock issue".to_string(),
        body: "This is a mock issue for testing".to_string(),
    };

    let proof = GithubApiProof {
        endpoint: url,
        timestamp: SystemTime::now(),
        status: 200,
    };

    Ok((issue, proof, budget))
}

/// Call LLM for analysis
pub fn call_llm(
    token: LlmToken,
    prompt: &str,
    budget: Budget,
) -> Result<(String, LlmCallProof, Budget)> {
    let budget = budget.spend(500)?; // LLM call costs 500 units

    // TODO: Real API call to Claude/GPT
    // For now, mock response
    let response = format!("Mock LLM response to: {}", prompt);

    let proof = LlmCallProof {
        model: token.model.clone(),
        timestamp: SystemTime::now(),
        tokens_used: 150,
    };

    Ok((response, proof, budget))
}

/// Data structures
#[derive(Debug, Clone)]
pub struct GitHubIssue {
    pub number: u32,
    pub title: String,
    pub body: String,
}

#[derive(Debug)]
pub struct Analysis {
    pub summary: String,
    pub proposed_fix: String,
}
```

---

## Part 2: Agent DSL

### Syntax Design

Keep it minimal. Focus on making linear patterns natural.

```
task <name>(
    <param>: <Type>,
    ...
) -> <ReturnType> {
    <statements>
}
```

**Key features:**
- All parameters are moved (consumed) by default
- Explicit `budget.spend(n)` for operations
- Question mark operator `?` for error handling
- Return statement required

**Example:**

```
task analyze_issue(
    github: GithubReadToken,
    filesystem: FileReadToken,
    llm: LlmToken,
    budget: Budget
) -> Analysis {
    // Fetch the issue (consumes github token, updates budget)
    let (issue, budget) = github.fetch_issue(123, budget)?;

    // Read relevant source files (consumes filesystem token)
    let (code, budget) = filesystem.read_file("src/main.rs", budget)?;

    // Ask LLM to analyze (consumes llm token)
    let prompt = format!("Analyze this issue: {}\n\nCode: {}", issue.body, code);
    let (analysis_text, budget) = llm.call(prompt, budget)?;

    // Return structured result
    return Analysis {
        summary: analysis_text,
        proposed_fix: "TODO: extract from analysis"
    };
}
```

### DSL Grammar (EBNF sketch)

```
program = task_def

task_def = "task" identifier "(" params ")" "->" type "{" statements "}"

params = param ("," param)*
param = identifier ":" type

statements = statement*
statement = let_binding | return_stmt

let_binding = "let" pattern "=" expr ";"
pattern = identifier | "(" identifier ("," identifier)* ")"

return_stmt = "return" expr ";"

expr = method_call | struct_literal | identifier | string_literal

method_call = expr "." identifier "(" args ")" "?"?
args = expr ("," expr)*

struct_literal = type "{" field_assigns "}"
field_assigns = field_assign ("," field_assign)*
field_assign = identifier ":" expr
```

### DSL Parser

Use `pest` for parsing. Create `src/dsl/grammar.pest`:

```pest
WHITESPACE = _{ " " | "\t" | "\n" | "\r" }
COMMENT = _{ "//" ~ (!"\n" ~ ANY)* }

program = { SOI ~ task_def ~ EOI }

task_def = { "task" ~ ident ~ "(" ~ params ~ ")" ~ "->" ~ type_name ~ block }

params = { param ~ ("," ~ param)* }
param = { ident ~ ":" ~ type_name }

block = { "{" ~ statement* ~ "}" }

statement = { let_stmt | return_stmt }

let_stmt = { "let" ~ pattern ~ "=" ~ expr ~ ";" }
pattern = { ident | tuple_pattern }
tuple_pattern = { "(" ~ ident ~ ("," ~ ident)* ~ ")" }

return_stmt = { "return" ~ expr ~ ";" }

expr = { method_call | struct_lit | ident | string_lit }

method_call = { primary ~ ("." ~ ident ~ "(" ~ args? ~ ")" ~ "?"?)+ }
primary = { ident | string_lit }
args = { expr ~ ("," ~ expr)* }

struct_lit = { type_name ~ "{" ~ field_assigns ~ "}" }
field_assigns = { field_assign ~ ("," ~ field_assign)* }
field_assign = { ident ~ ":" ~ expr }

type_name = @{ (ASCII_ALPHA ~ ASCII_ALPHANUMERIC*) }
ident = @{ ASCII_ALPHA ~ (ASCII_ALPHANUMERIC | "_")* }
string_lit = @{ "\"" ~ (!"\"" ~ ANY)* ~ "\"" }
```

### Code Generation

Parse DSL → Generate Rust code that uses the linear runtime.

```rust
// src/dsl/codegen.rs

pub fn generate_rust_code(ast: TaskDef) -> String {
    let mut output = String::new();

    // Generate function signature
    output.push_str(&format!("pub fn {}(\n", ast.name));
    for param in &ast.params {
        output.push_str(&format!("    {}: {},\n", param.name, param.type_name));
    }
    output.push_str(&format!(") -> Result<{}> {{\n", ast.return_type));

    // Generate body
    output.push_str("    let mut audit_trail = AuditTrail::new();\n");

    for stmt in &ast.statements {
        match stmt {
            Statement::Let { pattern, expr } => {
                output.push_str(&format!("    let {} = {};\n",
                    pattern_to_string(pattern),
                    expr_to_string(expr)
                ));
            }
            Statement::Return { expr } => {
                output.push_str(&format!("    Ok({})\n", expr_to_string(expr)));
            }
        }
    }

    output.push_str("}\n");
    output
}

// Helper functions for AST traversal
fn pattern_to_string(pattern: &Pattern) -> String {
    match pattern {
        Pattern::Ident(name) => name.clone(),
        Pattern::Tuple(names) => format!("({})", names.join(", ")),
    }
}

fn expr_to_string(expr: &Expr) -> String {
    match expr {
        Expr::Ident(name) => name.clone(),
        Expr::MethodCall { receiver, method, args, propagate_error } => {
            let args_str = args.iter().map(expr_to_string).collect::<Vec<_>>().join(", ");
            let call = format!("{}.{}({})",
                expr_to_string(receiver),
                method,
                args_str
            );
            if *propagate_error {
                format!("{}?", call)
            } else {
                call
            }
        }
        Expr::StructLit { type_name, fields } => {
            let fields_str = fields.iter()
                .map(|(k, v)| format!("{}: {}", k, expr_to_string(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{} {{ {} }}", type_name, fields_str)
        }
        Expr::StringLit(s) => format!("\"{}\"", s),
    }
}
```

---

## Part 3: LLM Integration

### Generation Prompt

```rust
// src/llm/generator.rs

pub fn generate_agent_prompt(task_description: &str) -> String {
    format!(r#"
You are generating code in a simple agent DSL. The DSL enforces linear resource usage.

AVAILABLE RESOURCES:
- github: GithubReadToken
  Methods: fetch_issue(issue_num, budget) -> (Issue, budget)

- filesystem: FileReadToken
  Methods: read_file(path, budget) -> (String, budget)

- llm: LlmToken
  Methods: call(prompt, budget) -> (String, budget)

- budget: Budget
  All operations consume budget and return updated budget

RULES:
1. Each token can only be used once (linear types)
2. Budget must be threaded through all operations
3. Use ? for error propagation
4. Must return the specified type

TASK: {}

Generate a task definition:

task solve(
    github: GithubReadToken,
    filesystem: FileReadToken,
    llm: LlmToken,
    budget: Budget
) -> Analysis {{
    // Your code here
}}

Only output the task definition, nothing else.
"#, task_description)
}
```

### Error Feedback Loop

```rust
// src/llm/repair.rs

pub struct RepairLoop {
    max_attempts: u32,
}

impl RepairLoop {
    pub fn new(max_attempts: u32) -> Self {
        RepairLoop { max_attempts }
    }

    pub async fn generate_and_compile(
        &self,
        task_description: &str,
        llm_client: &LlmClient,
    ) -> Result<CompiledAgent> {
        let mut attempt = 0;
        let mut last_code = String::new();

        loop {
            attempt += 1;
            if attempt > self.max_attempts {
                anyhow::bail!("Failed to generate valid code after {} attempts", self.max_attempts);
            }

            // Generate code
            let prompt = if attempt == 1 {
                generate_agent_prompt(task_description)
            } else {
                repair_prompt(&last_code, &last_error)
            };

            let code = llm_client.generate(&prompt).await?;
            last_code = code.clone();

            // Try to compile
            match compile_dsl(&code) {
                Ok(compiled) => return Ok(compiled),
                Err(e) => {
                    println!("Attempt {}: Compilation failed", attempt);
                    println!("Error: {}", e);
                    last_error = e;
                    // Loop continues to retry
                }
            }
        }
    }
}

fn repair_prompt(failed_code: &str, error: &CompileError) -> String {
    format!(r#"
The following code failed to compile:

```
{}
```

ERROR:
{}

COMMON ISSUES:
1. Forgetting to thread budget through operations
   Wrong: let (issue, _) = github.fetch_issue(123, budget)?;
   Right: let (issue, budget) = github.fetch_issue(123, budget)?;

2. Using a token twice (violates linearity)
   Wrong:
     let (file1, budget) = fs.read_file("a.rs", budget)?;
     let (file2, budget) = fs.read_file("b.rs", budget)?; // fs already consumed!
   Right: Use the token once

3. Not propagating errors with ?
   Wrong: let result = operation(token, budget);
   Right: let result = operation(token, budget)?;

Fix the code and output only the corrected task definition:
"#, failed_code, error)
}
```

### Compilation

```rust
// src/compiler.rs

use std::process::Command;
use std::fs;
use tempfile::TempDir;

pub struct CompiledAgent {
    pub executable_path: PathBuf,
    pub temp_dir: TempDir,
}

#[derive(Debug)]
pub struct CompileError {
    pub message: String,
    pub rust_errors: String,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}\n\nRust compiler output:\n{}", self.message, self.rust_errors)
    }
}

impl std::error::Error for CompileError {}

pub fn compile_dsl(dsl_code: &str) -> Result<CompiledAgent, CompileError> {
    // Parse DSL
    let ast = parse_dsl(dsl_code).map_err(|e| CompileError {
        message: format!("DSL parse error: {}", e),
        rust_errors: String::new(),
    })?;

    // Generate Rust code
    let rust_code = generate_rust_code(ast);

    // Create temporary project
    let temp_dir = TempDir::new().map_err(|e| CompileError {
        message: format!("Failed to create temp dir: {}", e),
        rust_errors: String::new(),
    })?;

    let project_path = temp_dir.path();

    // Write Cargo.toml
    let cargo_toml = r#"
[package]
name = "agent"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
linear-agent-runtime = { path = "../.." }
"#;
    fs::write(project_path.join("Cargo.toml"), cargo_toml).unwrap();

    // Write generated code
    fs::create_dir_all(project_path.join("src")).unwrap();
    let full_rust_code = format!(
        r#"
use linear_agent_runtime::*;
use anyhow::Result;

{}

fn main() {{
    println!("Agent compiled successfully");
}}
"#,
        rust_code
    );
    fs::write(project_path.join("src/main.rs"), full_rust_code).unwrap();

    // Compile with cargo
    let output = Command::new("cargo")
        .arg("build")
        .arg("--release")
        .current_dir(project_path)
        .output()
        .map_err(|e| CompileError {
            message: format!("Failed to run cargo: {}", e),
            rust_errors: String::new(),
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(CompileError {
            message: "Rust compilation failed".to_string(),
            rust_errors: stderr.to_string(),
        });
    }

    let executable_path = project_path.join("target/release/agent");

    Ok(CompiledAgent {
        executable_path,
        temp_dir,
    })
}

fn parse_dsl(code: &str) -> Result<TaskDef> {
    // Use pest parser
    let pairs = DslParser::parse(Rule::program, code)?;

    // Convert pest pairs to AST
    // TODO: Implement full AST construction

    Ok(TaskDef {
        name: "solve".to_string(),
        params: vec![],
        return_type: "Analysis".to_string(),
        statements: vec![],
    })
}
```

---

## Part 4: End-to-End Demo

### Main Application

```rust
// examples/github_issue_analyzer.rs

use linear_agent_runtime::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Linear Agent Runtime Demo ===\n");

    // Step 1: Define the task
    let task_description = r#"
Analyze GitHub issue #123 in the repository.
Read the main source file to understand the codebase.
Use the LLM to generate an analysis of the issue and propose a fix.
Return an Analysis struct with summary and proposed_fix fields.
"#;

    println!("Task: {}\n", task_description);

    // Step 2: LLM generates agent code
    println!("Generating agent code with LLM...");
    let llm_client = LlmClient::new(
        std::env::var("ANTHROPIC_API_KEY")?,
        "claude-3-5-sonnet-20241022"
    );

    let repair_loop = RepairLoop::new(5);
    let compiled_agent = repair_loop
        .generate_and_compile(task_description, &llm_client)
        .await?;

    println!("✓ Agent code generated and compiled\n");

    // Step 3: Create linear tokens with limited scope
    println!("Creating linear resource tokens...");
    let github_token = GithubReadToken::new(
        "owner/repo",
        std::env::var("GITHUB_TOKEN")?
    );
    let filesystem_token = FileReadToken::new("./src");
    let llm_token = LlmToken::new(
        std::env::var("ANTHROPIC_API_KEY")?,
        "claude-3-5-sonnet-20241022"
    );
    let budget = Budget::new(2000);

    println!("✓ Tokens created with scoped permissions\n");

    // Step 4: Execute agent with linear resources
    println!("Executing agent...");
    let audit_trail = AuditTrail::new();

    // In a real implementation, this would dynamically load and run the compiled agent
    // For demo purposes, we'll simulate it
    let result = simulate_agent_execution(
        github_token,
        filesystem_token,
        llm_token,
        budget,
        audit_trail,
    )?;

    println!("✓ Agent execution complete\n");

    // Step 5: Display results and audit trail
    println!("=== Results ===");
    println!("Summary: {}", result.analysis.summary);
    println!("Proposed Fix: {}", result.analysis.proposed_fix);
    println!();

    println!("=== Audit Trail ===");
    println!("{}", result.audit_trail.summary());
    println!();

    println!("=== Resource Usage ===");
    println!("Budget remaining: {} units", result.remaining_budget);

    Ok(())
}

struct ExecutionResult {
    analysis: Analysis,
    audit_trail: AuditTrail,
    remaining_budget: u32,
}

fn simulate_agent_execution(
    github: GithubReadToken,
    filesystem: FileReadToken,
    llm: LlmToken,
    budget: Budget,
    mut audit_trail: AuditTrail,
) -> Result<ExecutionResult> {
    // Fetch issue
    let (issue, proof, budget) = fetch_github_issue(github, 123, budget)?;
    audit_trail.record_github_call(proof);

    // Read source file
    let (code, proof, budget) = read_file(filesystem, "main.rs", budget)?;
    audit_trail.record_file_read(proof);

    // Call LLM for analysis
    let prompt = format!(
        "Analyze this GitHub issue and propose a fix.\n\nIssue: {}\n\nCode:\n{}",
        issue.body, code
    );
    let (analysis_text, proof, budget) = call_llm(llm, &prompt, budget)?;
    audit_trail.record_llm_call(proof);

    let analysis = Analysis {
        summary: analysis_text.clone(),
        proposed_fix: "Extract specific fix from analysis".to_string(),
    };

    Ok(ExecutionResult {
        analysis,
        audit_trail,
        remaining_budget: budget.remaining(),
    })
}
```

---

## Part 5: Testing Strategy

### Unit Tests for Linear Semantics

```rust
// tests/linearity_tests.rs

#[test]
fn test_budget_consumed_once() {
    let budget = Budget::new(100);
    let budget = budget.spend(50).unwrap();

    // This line would fail to compile if uncommented:
    // let _ = budget.spend(30); // Error: value used after move

    assert_eq!(budget.remaining(), 50);
}

#[test]
fn test_budget_exhaustion() {
    let budget = Budget::new(100);
    let result = budget.spend(150);

    assert!(result.is_err());
}

#[test]
fn test_token_cannot_be_cloned() {
    let token = FileReadToken::new("./test");

    // This would fail to compile:
    // let token2 = token.clone(); // Error: Clone not implemented

    // Token can only be used once
    let _ = token;
}

#[test]
fn test_budget_split() {
    let budget = Budget::new(1000);
    let (budget1, budget2) = budget.split();

    assert_eq!(budget1.remaining(), 500);
    assert_eq!(budget2.remaining(), 500);

    // Original budget is consumed, these would fail:
    // let _ = budget.spend(100); // Error: value used after move
}
```

### Integration Tests

```rust
// tests/dsl_compilation_tests.rs

#[test]
fn test_valid_dsl_compiles() {
    let dsl_code = r#"
task analyze(
    github: GithubReadToken,
    budget: Budget
) -> Analysis {
    let (issue, budget) = github.fetch_issue(123, budget)?;
    return Analysis {
        summary: "test",
        proposed_fix: "test"
    };
}
"#;

    let result = compile_dsl(dsl_code);
    assert!(result.is_ok());
}

#[test]
fn test_invalid_dsl_fails() {
    let dsl_code = r#"
task analyze(
    github: GithubReadToken,
    budget: Budget
) -> Analysis {
    let (issue, _) = github.fetch_issue(123, budget)?;
    let (issue2, _) = github.fetch_issue(456, budget)?; // Error: github used twice!
    return Analysis { summary: "test", proposed_fix: "test" };
}
"#;

    let result = compile_dsl(dsl_code);
    assert!(result.is_err());
}
```

---

## Part 6: Project Structure

```
linear-agent-runtime/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API exports
│   ├── tokens.rs           # Budget, FileReadToken, etc.
│   ├── proofs.rs           # Proof types and AuditTrail
│   ├── operations.rs       # read_file, fetch_github_issue, call_llm
│   ├── dsl/
│   │   ├── mod.rs
│   │   ├── grammar.pest    # Pest grammar
│   │   ├── parser.rs       # Pest parser wrapper
│   │   ├── ast.rs          # AST types
│   │   └── codegen.rs      # Rust code generation
│   ├── llm/
│   │   ├── mod.rs
│   │   ├── client.rs       # LLM API client
│   │   ├── generator.rs    # Prompt generation
│   │   └── repair.rs       # Error feedback loop
│   └── compiler.rs         # DSL -> Rust compilation
├── examples/
│   └── github_issue_analyzer.rs
└── tests/
    ├── linearity_tests.rs
    └── dsl_compilation_tests.rs
```

---

## Implementation Priorities

### Week 1: Core Runtime (CRITICAL)
- Implement `Budget`, `FileReadToken`, `GithubReadToken`, `LlmToken`
- Ensure no Clone/Copy derives
- Implement basic operations: `read_file`, `fetch_github_issue`, `call_llm`
- Write unit tests proving linearity

### Week 2: DSL Parser (HIGH)
- Define pest grammar
- Implement parser
- Build AST types
- Write parser tests

### Week 3: Code Generation (HIGH)
- Implement AST -> Rust codegen
- Handle budget threading automatically
- Generate compilation tests
- Verify generated code compiles

### Week 4: LLM Integration (MEDIUM)
- Implement LLM client (Claude API)
- Create generation prompts
- Build error feedback loop
- Test with real LLM

### Week 5: Demo & Polish (MEDIUM)
- Build end-to-end demo
- Add logging and debugging
- Write documentation
- Create example tasks

---

## Key Design Decisions

### Why Rust?
- Ownership system enforces linearity automatically via `!Clone` and `!Copy`
- No need to implement custom type checker
- Production-ready performance and safety

### Why a DSL instead of raw Rust?
- LLMs struggle with Rust's complexity
- DSL can enforce linear patterns by design
- Easier to generate correct code
- Can provide better error messages

### Why compile to Rust instead of interpret?
- Get full Rust type checking for free
- Compiler errors are precise and actionable
- Can feed errors back to LLM for repair
- Production performance

### Why pest for parsing?
- PEG parsers are simple and predictable
- Good error messages
- Easy to extend grammar
- Well-maintained library

---

## Success Criteria

After implementation, you should be able to:

1. **Create linear tokens** that cannot be cloned or reused
2. **Generate agent code** from natural language task descriptions
3. **Automatically compile** DSL to Rust with type checking
4. **Feed back compilation errors** to LLM for self-repair
5. **Execute agents** with provable resource constraints
6. **Produce audit trails** showing exactly what resources were used

### Demo Script

```
$ cargo run --example github_issue_analyzer

=== Linear Agent Runtime Demo ===

Task: Analyze GitHub issue #123...

Generating agent code with LLM...
Attempt 1: Compilation failed (budget not threaded)
Attempt 2: Success!
✓ Agent code generated and compiled

Creating linear resource tokens...
✓ Tokens created with scoped permissions

Executing agent...
✓ Agent execution complete

=== Results ===
Summary: The issue reports a null pointer exception...
Proposed Fix: Add null check before dereferencing...

=== Audit Trail ===
Files read: 1, GitHub calls: 1, LLM calls: 1

=== Resource Usage ===
Budget remaining: 1350 units
```

---

## Extension Points (Future Work)

- **More token types**: Database, AWS, Kubernetes, etc.
- **Parallel execution**: Budget splitting for concurrent sub-agents
- **Session types**: Multi-step protocols between agents
- **Gradual typing**: Mix linear and non-linear code with explicit boundaries
- **Visual debugger**: Show resource flow through agent execution
- **Marketplace**: Share verified agent templates

---

## Getting Started Checklist

- [ ] Create Rust project: `cargo new linear-agent-runtime --lib`
- [ ] Add dependencies: `anyhow`, `pest`, `pest_derive`, `tokio`, `reqwest`
- [ ] Implement `Budget` with linearity tests
- [ ] Implement `FileReadToken` and `read_file` operation
- [ ] Write first DSL example by hand
- [ ] Parse DSL with pest
- [ ] Generate Rust code from AST
- [ ] Compile generated code
- [ ] Integrate LLM for code generation
- [ ] Build error feedback loop
- [ ] Create end-to-end demo

**Start with Budget + one operation. Get that working end-to-end before adding complexity.**

## Appendix A: Complete Minimal Example

### src/lib.rs

```rust
// Re-export everything for easy use
pub mod tokens;
pub mod proofs;
pub mod operations;
pub mod dsl;
pub mod llm;
pub mod compiler;

pub use tokens::*;
pub use proofs::*;
pub use operations::*;
pub use compiler::{compile_dsl, CompileError, CompiledAgent};
pub use llm::{LlmClient, RepairLoop};
```

### Cargo.toml

```toml
[package]
name = "linear-agent-runtime"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
pest = "2.7"
pest_derive = "2.7"
tokio = { version = "1.35", features = ["full"] }
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tempfile = "3.8"

[dev-dependencies]
```

### AST Types (src/dsl/ast.rs)

```rust
#[derive(Debug, Clone)]
pub struct TaskDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: String,
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub type_name: String,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Let {
        pattern: Pattern,
        expr: Expr,
    },
    Return {
        expr: Expr,
    },
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Ident(String),
    Tuple(Vec<String>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Ident(String),
    StringLit(String),
    MethodCall {
        receiver: Box<Expr>,
        method: String,
        args: Vec<Expr>,
        propagate_error: bool,
    },
    StructLit {
        type_name: String,
        fields: Vec<(String, Expr)>,
    },
}
```

### Parser Implementation (src/dsl/parser.rs)

```rust
use pest::Parser;
use pest_derive::Parser;
use anyhow::Result;
use super::ast::*;

#[derive(Parser)]
#[grammar = "dsl/grammar.pest"]
pub struct DslParser;

pub fn parse_dsl(code: &str) -> Result<TaskDef> {
    let pairs = DslParser::parse(Rule::program, code)?;

    for pair in pairs {
        if pair.as_rule() == Rule::task_def {
            return parse_task_def(pair);
        }
    }

    anyhow::bail!("No task definition found")
}

fn parse_task_def(pair: pest::iterators::Pair<Rule>) -> Result<TaskDef> {
    let mut name = String::new();
    let mut params = Vec::new();
    let mut return_type = String::new();
    let mut statements = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => {
                if name.is_empty() {
                    name = inner.as_str().to_string();
                }
            }
            Rule::params => {
                params = parse_params(inner)?;
            }
            Rule::type_name => {
                return_type = inner.as_str().to_string();
            }
            Rule::block => {
                statements = parse_block(inner)?;
            }
            _ => {}
        }
    }

    Ok(TaskDef {
        name,
        params,
        return_type,
        statements,
    })
}

fn parse_params(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Param>> {
    let mut params = Vec::new();

    for param_pair in pair.into_inner() {
        if param_pair.as_rule() == Rule::param {
            let mut name = String::new();
            let mut type_name = String::new();

            for inner in param_pair.into_inner() {
                match inner.as_rule() {
                    Rule::ident => name = inner.as_str().to_string(),
                    Rule::type_name => type_name = inner.as_str().to_string(),
                    _ => {}
                }
            }

            params.push(Param { name, type_name });
        }
    }

    Ok(params)
}

fn parse_block(pair: pest::iterators::Pair<Rule>) -> Result<Vec<Statement>> {
    let mut statements = Vec::new();

    for stmt_pair in pair.into_inner() {
        match stmt_pair.as_rule() {
            Rule::let_stmt => {
                statements.push(parse_let_stmt(stmt_pair)?);
            }
            Rule::return_stmt => {
                statements.push(parse_return_stmt(stmt_pair)?);
            }
            _ => {}
        }
    }

    Ok(statements)
}

fn parse_let_stmt(pair: pest::iterators::Pair<Rule>) -> Result<Statement> {
    let mut pattern = Pattern::Ident(String::new());
    let mut expr = Expr::Ident(String::new());

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::pattern => {
                pattern = parse_pattern(inner)?;
            }
            Rule::expr => {
                expr = parse_expr(inner)?;
            }
            _ => {}
        }
    }

    Ok(Statement::Let { pattern, expr })
}

fn parse_return_stmt(pair: pest::iterators::Pair<Rule>) -> Result<Statement> {
    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::expr {
            return Ok(Statement::Return {
                expr: parse_expr(inner)?,
            });
        }
    }
    anyhow::bail!("Invalid return statement")
}

fn parse_pattern(pair: pest::iterators::Pair<Rule>) -> Result<Pattern> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => {
                return Ok(Pattern::Ident(inner.as_str().to_string()));
            }
            Rule::tuple_pattern => {
                let mut names = Vec::new();
                for ident in inner.into_inner() {
                    names.push(ident.as_str().to_string());
                }
                return Ok(Pattern::Tuple(names));
            }
            _ => {}
        }
    }
    anyhow::bail!("Invalid pattern")
}

fn parse_expr(pair: pest::iterators::Pair<Rule>) -> Result<Expr> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ident => {
                return Ok(Expr::Ident(inner.as_str().to_string()));
            }
            Rule::string_lit => {
                let s = inner.as_str();
                // Remove quotes
                return Ok(Expr::StringLit(s[1..s.len()-1].to_string()));
            }
            Rule::method_call => {
                return parse_method_call(inner);
            }
            Rule::struct_lit => {
                return parse_struct_lit(inner);
            }
            _ => {}
        }
    }
    anyhow::bail!("Invalid expression")
}

fn parse_method_call(pair: pest::iterators::Pair<Rule>) -> Result<Expr> {
    let mut receiver: Option<Expr> = None;
    let mut method = String::new();
    let mut args = Vec::new();
    let mut propagate_error = false;

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::primary => {
                receiver = Some(parse_expr(inner)?);
            }
            Rule::ident => {
                method = inner.as_str().to_string();
            }
            Rule::args => {
                for arg in inner.into_inner() {
                    args.push(parse_expr(arg)?);
                }
            }
            _ => {}
        }
    }

    // Check if ends with ?
    let text = pair.as_str();
    if text.ends_with('?') {
        propagate_error = true;
    }

    Ok(Expr::MethodCall {
        receiver: Box::new(receiver.unwrap()),
        method,
        args,
        propagate_error,
    })
}

fn parse_struct_lit(pair: pest::iterators::Pair<Rule>) -> Result<Expr> {
    let mut type_name = String::new();
    let mut fields = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::type_name => {
                type_name = inner.as_str().to_string();
            }
            Rule::field_assigns => {
                for field_assign in inner.into_inner() {
                    let mut name = String::new();
                    let mut expr = Expr::Ident(String::new());

                    for part in field_assign.into_inner() {
                        match part.as_rule() {
                            Rule::ident => name = part.as_str().to_string(),
                            Rule::expr => expr = parse_expr(part)?,
                            _ => {}
                        }
                    }

                    fields.push((name, expr));
                }
            }
            _ => {}
        }
    }

    Ok(Expr::StructLit { type_name, fields })
}
```

---

## Appendix B: Quick Start Commands

```bash
# Create project
cargo new linear-agent-runtime --lib
cd linear-agent-runtime

# Add to Cargo.toml dependencies
# (see Cargo.toml above)

# Create directory structure
mkdir -p src/dsl src/llm examples tests

# Create files
touch src/tokens.rs
touch src/proofs.rs
touch src/operations.rs
touch src/compiler.rs
touch src/dsl/mod.rs
touch src/dsl/grammar.pest
touch src/dsl/parser.rs
touch src/dsl/ast.rs
touch src/dsl/codegen.rs
touch src/llm/mod.rs
touch src/llm/client.rs
touch src/llm/generator.rs
touch src/llm/repair.rs
touch examples/github_issue_analyzer.rs
touch tests/linearity_tests.rs

# Test compilation
cargo build

# Run tests
cargo test

# Run example
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
cargo run --example github_issue_analyzer
```

---

## Appendix C: Expected LLM Output Example

**Input task:**
```
Analyze GitHub issue #42 about a login bug.
Read the auth.rs file to understand the authentication code.
Use the LLM to propose a fix.
```

**Expected generated DSL:**
```
task solve(
    github: GithubReadToken,
    filesystem: FileReadToken,
    llm: LlmToken,
    budget: Budget
) -> Analysis {
    let (issue, budget) = github.fetch_issue(42, budget)?;
    let (auth_code, budget) = filesystem.read_file("auth.rs", budget)?;

    let prompt = format("Issue: {}\n\nCode: {}\n\nPropose a fix.", issue.body, auth_code);
    let (fix_text, budget) = llm.call(prompt, budget)?;

    return Analysis {
        summary: issue.title,
        proposed_fix: fix_text
    };
}
```

**This compiles to Rust that:**
- Uses each token exactly once
- Threads budget through all operations
- Returns structured result
- Produces audit trail automatically

---

## Appendix D: Debugging Tips

### Common Compilation Errors

**Error: "value used after move"**
```
let (data, budget) = read_file(fs_token, "test.txt", budget)?;
let (more, budget) = read_file(fs_token, "other.txt", budget)?;
                                ^^^^^^^^ value used here after move
```
**Fix:** Each token can only be used once. Use a different approach or split tokens.

**Error: "budget not threaded"**
```
let (issue, _) = github.fetch_issue(123, budget)?;
let (file, _) = filesystem.read_file("main.rs", budget)?;
                                                 ^^^^^^ value used after move
```
**Fix:** Thread budget through: `let (issue, budget) = ...`

**Error: "missing return statement"**
```
task solve(...) -> Analysis {
    let (result, budget) = do_work(...)?;
    Analysis { summary: result }  // Missing 'return'
}
```
**Fix:** Add `return` keyword.

### Testing Linearity

```rust
// This test should NOT compile (uncomment to verify)
#[test]
#[ignore]
fn test_cannot_reuse_token() {
    let token = FileReadToken::new("./test");
    let _ = read_file(token, "a.txt", Budget::new(100));
    // let _ = read_file(token, "b.txt", Budget::new(100)); // Should fail
}
```

---

## Appendix E: Metrics to Track

### Development Metrics
- Lines of DSL code LLM can generate correctly on first try
- Average number of repair iterations needed
- Percentage of tasks that compile within 5 attempts
- Types of errors most common (for improving prompts)

### Runtime Metrics
- Budget consumption per operation type
- Audit trail completeness (all operations logged)
- Resource leak detection (tokens not consumed)
- Execution time overhead vs. direct Rust

### Security Metrics
- Attempted privilege escalations (caught by type system)
- Path traversal attempts (blocked)
- Token reuse attempts (compilation failures)
- Budget violations (runtime errors)

---

## Final Notes

This design gives you:

✅ **Provable resource safety** through Rust's type system
✅ **LLM-friendly DSL** that's easier to generate than raw Rust
