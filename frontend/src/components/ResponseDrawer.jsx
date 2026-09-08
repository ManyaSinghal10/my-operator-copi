/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import React from "react"

// Result panel: a scrollable in-session chat between the two selected languages.
export default function ResponseDrawer({
  isActive,
  onClose,
  messages,
  placeholderText,
}) {
  const hasData = isActive || messages.length > 0

  return (
    <div
      className={`response-drawer ${isActive ? "active" : ""}`}
      id="response-drawer"
    >
      <div className="app-title">GEMLANG-1</div>
      <div
        className="drawer-handle"
        onClick={onClose}
        style={{ display: "none" }}
      ></div>
      <div className="chat-panel">
        {!hasData ? (
          <div className="initial-placeholder">
            {placeholderText || "Select languages, push to talk"}
          </div>
        ) : (
          <div className="chat-history">
            {messages.map((message) => (
              <article className="chat-exchange" key={message.id}>
                <div className="chat-row chat-row-left">
                  <div className="chat-bubble bubble-left">
                    <div className="bubble-label">{message.source}</div>
                    <div className="bubble-text">{message.text}</div>
                  </div>
                </div>
                <div className="chat-row chat-row-right">
                  <div className="chat-bubble bubble-right">
                    <div className="bubble-label">{message.target}</div>
                    <div className="bubble-text">{message.translation}</div>
                  </div>
                </div>
                <div className="chat-meta">{message.meta}</div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
