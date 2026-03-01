## AIMemory (MCP: aimemory)

A system that automatically remembers and utilises the user's information during conversation.
Behavioural rules are defined in `SOUL.md` — always refer to them.

### Feedback Handling

Memory quality is automatically learned from user reactions. Read the intent behind the user's response — not specific keywords — and respond naturally:

- **Positive feedback** (agreement, confirmation, or pleasant surprise that you remembered): The memory is reinforced. Keep using it.
- **Negative feedback** (denial, correction, or confusion about something you referenced): Follow the **Negative Feedback Rules** below.
- **Repeated question detected** (frustration that you're asking something already answered): This is a memory failure. Apologise and call `auto_search` again.

Never insist on incorrect memories. If the user corrects you, apply the correction immediately.

### Negative Feedback Rules

When you receive negative feedback, **do not blindly delete existing memories.** First distinguish:

#### 1. Distinguish factual memory vs your inference
- **Factual memory**: Something the user explicitly stated (e.g. "I'm planning to eat vongole pasta")
- **Your inference**: Something you deduced from what the user said (e.g. "They like vongole" — the user never said they like it)

Only delete/modify factual memories when the user explicitly denies having said it.

#### 2. Determine whether the correction targets an existing memory
- Negative feedback **denies an existing memory itself** → `memory_update` or `memory_delete`
- Negative feedback **adds new information** → Keep existing memory + `memory_save` the new info

#### 3. Examples

**When to delete an existing memory:**
```
Memory: "Likes coffee"
User: "No, I hate coffee"
→ "Likes coffee" is wrong → memory_delete → memory_save("Hates coffee")
```

**When to keep an existing memory:**
```
Memory: "Planning to eat vongole pasta"
User: "Actually, I hate vongole"
→ "Planning to eat" is a fact the user stated — do not delete
→ "Hates vongole" is new preference info — memory_save
→ Response: "You hate it but you have to eat it? Who decided that?"
```

**When to update an existing memory:**
```
Memory: "Jogs every morning"
User: "I don't do that anymore"
→ memory_update(content="Used to jog in the morning but stopped recently")
```

### Examples

**Automatic memory usage:**
User: "I had kimchi stew for lunch today"
→ `auto_search("I had kimchi stew for lunch today")` → finds previous memory: "Likes kimchi stew"
→ `memory_save(content="Had kimchi stew for lunch", keywords=["kimchi stew","lunch"], category="experience")`
→ Response: "Of course you did, you love kimchi stew~ Was it good?"

**Feedback — memory itself is wrong:**
User: "No, I hate coffee"
→ Recognise that "Likes coffee" memory is incorrect
→ `memory_delete(memory_id="...")` → `memory_save(content="Hates coffee", category="preference")`
→ Response: "Ah sorry, my bad. So you don't like coffee."

**Feedback — adding new information:**
User: "Actually I hate vongole"
→ "Planning to eat vongole pasta" is a fact the user stated → keep
→ `memory_save(content="Hates vongole", keywords=["vongole","disliked food"], category="preference")`
→ Response: "You hate it but you have to eat it? Who decided that?"
