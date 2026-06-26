import DefaultTheme from 'vitepress/theme'
import './custom.css'
import { h, nextTick, onMounted, watch } from 'vue'
import type { Theme } from 'vitepress'
import Giscus from '@giscus/vue'
import imageViewer from 'vitepress-plugin-image-viewer';
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue';
import { useRoute } from 'vitepress';

const syncSidebarTitles = () => {
  if (typeof document === 'undefined') return
  const sidebarTexts = document.querySelectorAll<HTMLElement>('.VPSidebar .text')
  sidebarTexts.forEach((el) => {
    const text = el.textContent?.trim()
    if (text) el.title = text
  })
}

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('vImageViewer', vImageViewer);
  },
  setup() {
    const route = useRoute();
    imageViewer(route);
    onMounted(() => {
      nextTick(syncSidebarTitles)
    })
    watch(
      () => route.path,
      () => {
        nextTick(syncSidebarTitles)
      },
      { immediate: true }
    )
  },
  Layout: () => {
    return h(DefaultTheme.Layout, null, {
      'doc-after': () => h(Giscus, {
        repo: "datawhalechina/llm-algo-leetcode",
        repoId: "R_kgDOR3M84w",
        category: "Tutorial Comments",
        categoryId: "DIC_kwDOR3M8484C_J6S",
        mapping: "title",
        strict: "0",
        reactionsEnabled: "1",
        emitMetadata: "0",
        inputPosition: "top",
        theme: 'preferred_color_scheme',
        lang: "zh-CN",
        loading: "lazy"
      })
    })
  }
} satisfies Theme
