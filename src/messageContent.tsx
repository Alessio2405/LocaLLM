import { ReactNode, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { Role } from "./types";

const STRUCTURED_MESSAGE_BLOCK =
  /^(\s*[-*+]\s|\s*\d+\.\s|\s*>\s|\s*#{1,6}\s|\s*\|.*\||\s*[-*_]{3,}\s*$)/m;
const CODE_BLOCK_PATTERN = /(```[\s\S]*?```|~~~[\s\S]*?~~~)/g;
const THINK_BLOCK_PATTERN = /<think\b[^>]*>[\s\S]*?<\/think>/gi;
const THINK_TAG_PATTERN = /<\/?think\b[^>]*>/gi;
const CHAT_TEMPLATE_TAIL_PATTERN =
  /(?:<\|im_start\|>\s*(?:system|user|assistant)\b|<\|start_header_id\|>\s*(?:system|user|assistant)\s*<\|end_header_id\|>)[\s\S]*$/gi;
const SPECIAL_TOKEN_PATTERN =
  /<\|(?:im_start|im_end|endoftext|eot_id|start_header_id|end_header_id)[^|>]*\|>/gi;
const INSTRUCTION_TOKEN_PATTERN = /\[\/?INST\]|<<\/?SYS>>|<s>|<\/s>/gi;

function formatAssistantMarkdown(text: string) {
  return text
    .replace(/([.!?])([A-ZÀ-ÖØ-Þ])/g, "$1 $2")
    .replace(/([.!?:])([*+-]\s+)/g, "$1\n\n$2")
    .replace(/([.!?:])(\d+\.\s+)/g, "$1\n\n$2")
    .replace(/([.!?:])(\(Nota:)/g, "$1\n\n$2")
    .replace(/\n{3,}/g, "\n\n");
}

export function stripAssistantInternalContent(content: string) {
  return content
    .split(CODE_BLOCK_PATTERN)
    .map((segment) => {
      const trimmed = segment.trimStart();
      if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
        return segment;
      }

      const withoutClosedThinkBlocks = segment.replace(THINK_BLOCK_PATTERN, "");
      const danglingThinkIndex = withoutClosedThinkBlocks.toLowerCase().lastIndexOf("<think");
      const withoutDanglingThinkBlock =
        danglingThinkIndex >= 0
          ? withoutClosedThinkBlocks.slice(0, danglingThinkIndex)
          : withoutClosedThinkBlocks;

      return formatAssistantMarkdown(
        withoutDanglingThinkBlock
          .replace(CHAT_TEMPLATE_TAIL_PATTERN, "")
          .replace(THINK_TAG_PATTERN, "")
          .replace(SPECIAL_TOKEN_PATTERN, "")
          .replace(INSTRUCTION_TOKEN_PATTERN, "")
          .replace(/\n{3,}/g, "\n\n"),
      );
    })
    .join("")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function isStructuredMessageBlock(block: string) {
  return STRUCTURED_MESSAGE_BLOCK.test(block);
}

function normalizeStructuredBlock(block: string) {
  return block
    .split("\n")
    .map((line) => line.replace(/[ \t]+$/g, ""))
    .filter((line) => line.trim().length > 0)
    .join("\n")
    .trim();
}

function normalizeFreeformBlock(block: string) {
  return block
    .split("\n")
    .map((line) => line.trim().replace(/[ \t]+/g, " "))
    .filter(Boolean)
    .join(" ")
    .trim();
}

function normalizeAssistantSegment(segment: string) {
  return segment
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) {
        return "";
      }
      return isStructuredMessageBlock(trimmed)
        ? normalizeStructuredBlock(trimmed)
        : normalizeFreeformBlock(trimmed);
    })
    .filter(Boolean)
    .join("\n\n");
}

function normalizeAssistantMessage(content: string) {
  return stripAssistantInternalContent(content)
    .replace(/\r\n/g, "\n")
    .replace(/\u00c2/g, "")
    .replace(/\u00a0/g, " ")
    .split(CODE_BLOCK_PATTERN)
    .map((segment) => {
      const trimmed = segment.trim();
      if (!trimmed) {
        return "";
      }
      return trimmed.startsWith("```") || trimmed.startsWith("~~~")
        ? trimmed
        : normalizeAssistantSegment(trimmed);
    })
    .filter(Boolean)
    .join("\n\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export function getAssistantMessagePreview(content: string) {
  return normalizeAssistantMessage(content)
    .replace(CODE_BLOCK_PATTERN, " [code block] ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeUserMessage(content: string) {
  return content
    .replace(/\r\n/g, "\n")
    .replace(/\u00c2/g, "")
    .replace(/\u00a0/g, " ");
}

function flattenNodeText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(flattenNodeText).join("");
  }
  if (!node || typeof node !== "object") {
    return "";
  }
  const maybeChildren = (node as { props?: { children?: ReactNode } }).props?.children;
  return flattenNodeText(maybeChildren);
}

function CodeBlock({ children }: { children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = useMemo(() => flattenNodeText(children).replace(/\n$/, ""), [children]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div className="message-code-shell">
      <button className="code-copy-button" type="button" onClick={() => void handleCopy()}>
        {copied ? "Copied" : "Copy"}
      </button>
      <pre>{children}</pre>
    </div>
  );
}

const markdownComponents: Components = {
  a: ({ ...props }) => <a {...props} target="_blank" rel="noreferrer" />,
  pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
  table: ({ children }) => (
    <div className="message-table-wrap">
      <table>{children}</table>
    </div>
  ),
};

type MessageContentProps = {
  role: Role;
  content: string;
};

export function MessageContent({ role, content }: MessageContentProps) {
  if (role === "user") {
    return <p className="message-paragraph user-copy">{normalizeUserMessage(content)}</p>;
  }

  const normalized = normalizeAssistantMessage(content);
  return (
    <div className="message-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
        {normalized}
      </ReactMarkdown>
    </div>
  );
}
