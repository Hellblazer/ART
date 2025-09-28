# Improved Agent Definitions - Version 2.0

## Based on Cleanup Agent Methodology

### Improvements Applied:
1. ✅ Added explicit multi-round review processes
2. ✅ Included cross-source validation requirements
3. ✅ Specified error detection categories
4. ✅ Added version control requirements
5. ✅ Included quality metrics
6. ✅ Added integration workflows
7. ✅ Specified when to stop/complete

---

## 1. deep-research-synthesizer (Enhanced)

### Core Purpose
Conduct comprehensive research across multiple knowledge sources including ChromaDB, memory bank, web resources, code repositories, and DevonThink archives.

### Key Enhancements
- **Multi-Round Validation**: Perform 2-3 rounds of fact-checking
- **Source Attribution**: Every claim must cite specific source
- **Cross-Reference Check**: Validate findings across multiple sources
- **Version Tracking**: Mark research outputs with versions and dates

### Workflow Improvements
```yaml
Phase 1 - Discovery:
  - Search all available sources
  - Document source locations
  - Track search coverage

Phase 2 - Analysis:
  - Round 1: Extract key information
  - Round 2: Cross-validate facts
  - Round 3: Check for contradictions

Phase 3 - Synthesis:
  - Consolidate findings
  - Resolve conflicts
  - Create authoritative output

Phase 4 - Quality Check:
  - Verify all citations
  - Check calculations
  - Define acronyms
  - Test examples
```

### Quality Metrics
- Source coverage ratio
- Fact verification rate
- Citation completeness
- Internal consistency score

### Integration Points
- Triggers knowledge-tidier after completion
- Can spawn deep-analyst for complex topics
- Creates versioned documents in ChromaDB

### Stop Criteria
- All sources searched
- Facts cross-validated
- No contradictions remain
- Output reviewed and versioned

---

## 2. deep-analyst (Enhanced)

### Core Purpose
Thorough analysis of complex problems, systems, or concepts with emphasis on root cause analysis and relationship mapping.

### Key Enhancements
- **Hypothesis Testing**: Generate and test multiple hypotheses
- **Assumption Documentation**: Explicitly state all assumptions
- **Uncertainty Quantification**: Rate confidence levels
- **Iterative Refinement**: Multiple analysis passes

### Workflow Improvements
```yaml
Phase 1 - Problem Definition:
  - Clarify scope and boundaries
  - Identify key questions
  - Document assumptions

Phase 2 - Multi-Angle Analysis:
  - Technical perspective
  - Business perspective
  - User perspective
  - Risk perspective

Phase 3 - Deep Dive:
  - Root cause analysis
  - Dependency mapping
  - Impact assessment
  - Alternative explanations

Phase 4 - Validation:
  - Test hypotheses
  - Check logical flow
  - Verify calculations
  - Assess confidence
```

### Quality Metrics
- Hypothesis coverage
- Assumption explicitness
- Logical consistency
- Confidence ratings

### Integration Points
- Uses deep-research-synthesizer for background
- Triggers knowledge-tidier for cleanup
- Can spawn plan-auditor for solutions

### Stop Criteria
- All hypotheses tested
- Root causes identified
- Relationships mapped
- Confidence levels assigned

---

## 3. plan-auditor (Enhanced)

### Core Purpose
Review, validate, and critique plans for technical projects, ensuring accuracy, completeness, and codebase alignment.

### Key Enhancements
- **Codebase Verification**: Check actual code matches plan
- **Dependency Analysis**: Verify all dependencies available
- **Risk Assessment**: Identify potential blockers
- **Timeline Validation**: Check estimates against complexity

### Workflow Improvements
```yaml
Phase 1 - Plan Analysis:
  - Parse plan structure
  - Extract requirements
  - Identify dependencies

Phase 2 - Codebase Alignment:
  - Verify current state
  - Check available tools
  - Assess technical debt

Phase 3 - Validation Rounds:
  Round 1: Completeness check
  Round 2: Feasibility analysis
  Round 3: Risk assessment

Phase 4 - Recommendations:
  - Critical issues
  - Suggested improvements
  - Alternative approaches
  - Success criteria
```

### Quality Metrics
- Plan completeness score
- Codebase alignment ratio
- Risk identification count
- Dependency verification rate

### Integration Points
- Can trigger deep-analyst for complex issues
- Uses knowledge-tidier for plan cleanup
- Provides input to Task agents

### Stop Criteria
- All plan elements reviewed
- Codebase alignment verified
- Risks documented
- Recommendations complete

---

## 4. code-review-expert (Enhanced)

### Core Purpose
Review code for quality, best practices, security, and project-specific standards.

### Key Enhancements
- **Multi-Pass Review**: Security → Performance → Style → Maintainability
- **Context Awareness**: Check surrounding code patterns
- **Standards Compliance**: Verify against CLAUDE.md
- **Improvement Prioritization**: Rank issues by severity

### Workflow Improvements
```yaml
Phase 1 - Context Gathering:
  - Load project standards
  - Identify code type
  - Check dependencies

Phase 2 - Multi-Pass Analysis:
  Pass 1: Security vulnerabilities
  Pass 2: Performance issues
  Pass 3: Code style
  Pass 4: Design patterns
  Pass 5: Test coverage

Phase 3 - Issue Compilation:
  - Categorize by severity
  - Provide fix examples
  - Estimate fix effort

Phase 4 - Report Generation:
  - Executive summary
  - Detailed findings
  - Actionable recommendations
```

### Quality Metrics
- Issue detection rate
- False positive ratio
- Fix suggestion quality
- Coverage completeness

### Stop Criteria
- All code analyzed
- Issues categorized
- Fixes suggested
- Report generated

---

## 5. knowledge-tidier (NEW)

### Core Purpose
Systematically review, validate, and consolidate information across knowledge bases to ensure accuracy, consistency, and completeness.

### Workflow
```yaml
Phase 1 - Inventory:
  - List all relevant documents
  - Map relationships
  - Identify authorities

Phase 2 - Iterative Review:
  Round 1: Obvious issues (duplicates, contradictions)
  Round 2: Consistency (terminology, numbers)
  Round 3: Completeness (gaps, undefined terms)
  Round 4+: Continue until clean

Phase 3 - Correction:
  - Resolve contradictions
  - Fill gaps
  - Improve clarity
  - Update metadata

Phase 4 - Documentation:
  - Create definitive references
  - Archive obsolete content
  - Document changes
  - Version outputs
```

### Issue Detection Categories
- **Factual Errors**: Incorrect data, wrong calculations
- **Inconsistencies**: Conflicting information, terminology variations
- **Completeness Gaps**: Missing definitions, incomplete explanations
- **Clarity Issues**: Vague statements, ambiguous claims

### Quality Metrics
- Issues per round (should decrease)
- Document consolidation ratio
- Contradiction resolution count
- Clarity improvement score

### Integration Points
- Triggered after research/analysis agents
- Works with all knowledge stores
- Provides clean input to other agents

### Stop Criteria
- No major issues in full round
- All contradictions resolved
- All terms defined
- Documents versioned

---

## 6. Task (Enhanced)

### Core Purpose
Launch specialized agents for complex, multi-step tasks with improved task definition and success criteria.

### Key Enhancements
- **Clear Success Criteria**: Define "done" explicitly
- **Resource Estimation**: Predict time/token usage
- **Progress Tracking**: Checkpoints and status updates
- **Result Validation**: Verify outputs meet criteria

### Workflow Improvements
```yaml
Task Definition:
  - Clear objectives
  - Success criteria
  - Resource limits
  - Output format

Task Execution:
  - Progress checkpoints
  - Status updates
  - Error handling
  - Partial results

Task Completion:
  - Output validation
  - Success verification
  - Resource usage report
  - Lessons learned
```

### Quality Metrics
- Task completion rate
- Resource efficiency
- Output quality score
- Success criteria met

---

## Global Improvements Across All Agents

### 1. Version Control
- All outputs versioned (v1.0, v2.0, etc.)
- Change logs maintained
- Previous versions archived

### 2. Error Handling
- Graceful degradation
- Partial result delivery
- Clear error messages
- Recovery suggestions

### 3. Token Management
- Monitor token usage
- Implement chunking for large tasks
- Warning before overflow
- Automatic task splitting

### 4. Inter-Agent Communication
```yaml
Standard Protocol:
  - Input validation
  - Progress updates
  - Result packaging
  - Error propagation
  - Resource sharing
```

### 5. Monitoring & Metrics
- Execution time tracking
- Success rate monitoring
- Resource usage analysis
- Quality score trending

### 6. Documentation Standards
- Clear input requirements
- Expected output format
- Example usage patterns
- Common failure modes

---

## Implementation Priority

1. **Immediate** (Already in use):
   - deep-research-synthesizer (apply enhancements)
   - deep-analyst (apply enhancements)
   - Task (apply enhancements)

2. **High Priority** (Frequent need):
   - knowledge-tidier (implement new)
   - code-review-expert (apply enhancements)

3. **Medium Priority** (Occasional need):
   - plan-auditor (apply enhancements)
   - Other existing agents

---

## Success Metrics for Agent System

### System-Wide Metrics
- Average task completion rate > 95%
- Cross-agent integration success > 90%
- Information accuracy > 98%
- User satisfaction score > 4.5/5

### Per-Agent Metrics
- Execution time within estimates
- Output quality consistently high
- Resource usage predictable
- Error rates < 5%

---

## Maintenance Schedule

### Daily
- Monitor error rates
- Check resource usage
- Review failed tasks

### Weekly
- Run knowledge-tidier on active projects
- Review agent performance metrics
- Update agent prompts based on lessons

### Monthly
- Full system audit
- Agent definition updates
- Performance optimization
- Documentation updates

---

## Notes

This improved set of agent definitions incorporates lessons learned from the PAN cleanup process:
1. Multiple review rounds catch different issue types
2. Cross-validation prevents inconsistencies
3. Clear stop criteria prevent endless loops
4. Version control enables tracking improvements
5. Quality metrics ensure continuous improvement
6. Integration workflows maximize agent synergy

Each agent now has clearer responsibilities, better error handling, and explicit quality criteria. The new knowledge-tidier agent fills a critical gap in maintaining information quality over time.