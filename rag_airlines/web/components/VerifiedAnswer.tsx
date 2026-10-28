'use client';

import {memo} from 'react';
import Markdown from 'react-markdown';
import type {Root, Parents, Text} from 'mdast';

// Parse the complete, released answer once. Character opacity changes keep the
// final Markdown layout stable; no partial Markdown or model tokens are rendered.
function revealCharacters() {
  return (tree: Root) => {
    const texts: Text[] = [];
    function collect(parent: Parents) {
      for (const node of parent.children) {
        if (node.type === 'text') texts.push(node);
        else if ('children' in node) collect(node);
      }
    }
    collect(tree);
    const count = texts.reduce((total, text) => total + Array.from(text.value).length, 0);
    let index = 0;
    for (const text of texts) {
      text.data = {...text.data, hName: 'span', hChildren: Array.from(text.value).map(character => ({
        type: 'element', tagName: 'span',
        properties: {className: ['reveal-character'], style: `--reveal-at:${index++ / Math.max(count, 1)}`},
        children: [{type: 'text', value: character}],
      }))};
    }
  };
}

const markdownComponents = {
  // Source navigation uses the dedicated citation controls. Model-generated
  // links stay readable without introducing hidden, focusable link duplicates.
  a: ({children}: {children?: React.ReactNode}) => <span>{children}</span>,
};
const excluded = ['img', 'iframe', 'script', 'style', 'input', 'button'];

export const VerifiedAnswer = memo(function VerifiedAnswer({text}: {text: string}) {
  return <div className="answer-content answer-markdown">
    <div className="answer-visual">
      <Markdown skipHtml disallowedElements={excluded} components={markdownComponents} remarkPlugins={[revealCharacters]}>{text}</Markdown>
    </div>
  </div>;
});
