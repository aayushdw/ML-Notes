import { QuartzConfig } from "./quartz/cfg"
import * as Plugin from "./quartz/plugins"

/**
 * Quartz 4 Configuration
 *
 * See https://quartz.jzhao.xyz/configuration for more information.
 */
const config: QuartzConfig = {
  configuration: {
    pageTitle: "Aayush's ML & AI Notes",
    pageTitleSuffix: "",
    enableSPA: true,
    enablePopovers: true,
    analytics: null,
    locale: "en-US",
    baseUrl: "aayushdw.github.io/ML-Notes",
    ignorePatterns: ["private", "templates", ".obsidian", "CLAUDE.md", "GEMINI.md"],
    defaultDateType: "modified",
    theme: {
      fontOrigin: "googleFonts",
      cdnCaching: true,
      typography: {
        header: "Schibsted Grotesk",
        body: "Source Sans Pro",
        code: "IBM Plex Mono",
      },
      colors: {
        // Retroma Groovy Theme - Teal Accent
        lightMode: {
          light: "#f5f5f0",
          lightgray: "#d8d8d0",
          gray: "#8a8a80",
          darkgray: "#3a3a36",
          dark: "#1a1a18",
          secondary: "#0d7377",
          tertiary: "#14919b",
          highlight: "rgba(13, 115, 119, 0.12)",
          textHighlight: "#b8e0e388",
        },
        darkMode: {
          light: "#1a1a1c",
          lightgray: "#2e2e30",
          gray: "#5a5a5c",
          darkgray: "#d0d0ce",
          dark: "#f0f0ec",
          secondary: "#2dd4bf",
          tertiary: "#5eead4",
          highlight: "rgba(45, 212, 191, 0.15)",
          textHighlight: "#0d737766",
        },
      },
    },
  },
  plugins: {
    transformers: [
      Plugin.FrontMatter(),
      Plugin.CreatedModifiedDate({
        priority: ["frontmatter", "git", "filesystem"],
      }),
      Plugin.SyntaxHighlighting({
        theme: {
          light: "github-light",
          dark: "github-dark",
        },
        keepBackground: false,
      }),
      Plugin.ObsidianFlavoredMarkdown({ enableInHtmlEmbed: false }),
      Plugin.GitHubFlavoredMarkdown(),
      Plugin.TableOfContents(),
      Plugin.CrawlLinks({ markdownLinkResolution: "shortest" }),
      Plugin.Description(),
      Plugin.Latex({ renderEngine: "katex" }),
    ],
    filters: [Plugin.RemoveDrafts()],
    emitters: [
      Plugin.AliasRedirects(),
      Plugin.ComponentResources(),
      Plugin.ContentPage(),
      Plugin.FolderPage(),
      Plugin.TagPage(),
      Plugin.ContentIndex({
        enableSiteMap: true,
        enableRSS: true,
      }),
      Plugin.Assets(),
      Plugin.Static(),
      Plugin.Favicon(),
      Plugin.NotFoundPage(),
      Plugin.CustomOgImages(),
    ],
  },
}

export default config
