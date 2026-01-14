To address a Gemini review, run the following workflow:

- Get comments from the pull request from Gemini
- find only recently added, unresolved ones
- output them to me
- address them: read the comment and suggested code changes, make the changes
- tell me what was changed, per comment and how it addressed the feedback
- commit the changes, if there are multiple changes, create multiple commits
- push the changes to the branch
- trigger another review with comment "/gemini review".