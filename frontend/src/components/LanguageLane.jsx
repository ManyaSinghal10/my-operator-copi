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

import React, { useEffect, useRef } from "react"

// One person's lane: the vertical "revolver" language selector plus the
// recording highlight state.
export default function LanguageLane({
  laneId,
  laneLabel,
  languages,
  currentIndex,
  isRecording,
  isActivePerson,
  onLanguageChange,
  onRotate,
  onRecordStart,
  onRecordStop,
}) {
  const drumRef = useRef(null)

  // Slide the drum so the current language row is in the viewport.
  // The 44px row height must match .revolver-item in style.css.
  useEffect(() => {
    if (drumRef.current) {
      drumRef.current.style.transform = `translateY(-${currentIndex * 44}px)`
    }
  }, [currentIndex])

  const handlePrev = (e) => {
    e.preventDefault()
    if (!isRecording) onRotate(-1)
  }

  const handleNext = (e) => {
    e.preventDefault()
    if (!isRecording) onRotate(1)
  }

  const handleRecordClick = () => {
    if (isRecording) {
      onRecordStop()
    } else {
      onRecordStart(laneId)
    }
  }

  return (
    <section
      className={`language-lane lane-${laneId === 1 ? "one" : "two"} ${isRecording ? "recording" : ""} ${isActivePerson ? "selected" : ""}`}
      id={`lane-${laneId}`}
    >
      <div className="lane-header">
        <span className="lane-label">{laneLabel}</span>
      </div>
      <div className={`revolver-stage ${laneId === 1 ? "source-stage" : ""}`}>
        {laneId === 1 ? (
          <div className="source-language-control">
            <select
              className="source-language-select"
              aria-label="Source language"
              value={currentIndex}
              disabled={isRecording}
              onChange={(event) => onLanguageChange(event.target.value)}
            >
              {languages.map((language, index) => (
                <option key={language.code} value={index}>
                  {language.name}
                </option>
              ))}
            </select>
            <span className="source-language-chevron" aria-hidden="true">
              ▾
            </span>
          </div>
        ) : (
          <div className="revolver-viewport">
            <div className="revolver-drum" ref={drumRef}>
              {languages.map((l) => (
                <div key={l.code} className="revolver-item">
                  {l.name}
                </div>
              ))}
            </div>
          </div>
        )}
        {laneId === 1 && (
          <>
            <button
              className={`record-button ${isRecording ? "recording" : ""}`}
              type="button"
              aria-label={`${isRecording ? "Stop" : "Start"} recording for lane ${laneLabel}`}
              onClick={handleRecordClick}
            >
              <svg
                className="microphone-icon"
                viewBox="0 0 24 24"
                aria-hidden="true"
                focusable="false"
              >
                <path d="M12 15.5a3.5 3.5 0 0 0 3.5-3.5V6a3.5 3.5 0 1 0-7 0v6a3.5 3.5 0 0 0 3.5 3.5Z" />
                <path d="M18.5 11.5a6.5 6.5 0 0 1-13 0M12 18v3M8.5 21h7" />
              </svg>
            </button>
          </>
        )}
      </div>
      <span className="language-name" style={{ display: "none" }}>
        {languages[currentIndex].name}
      </span>
      <div className="corner-brackets"></div>
    </section>
  )
}
