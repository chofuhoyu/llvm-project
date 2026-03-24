// 高级示例：展示如何扩展#pragma learn功能
// 这个文件不是要编译的，而是作为参考代码

// ============================================
// 示例1：使用AST Consumer实现printf插入
// ============================================

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

// 存储#pragma learn print位置的信息
struct LearnPrintLocation {
    SourceLocation Loc;
    std::string Message;
};

static std::vector<LearnPrintLocation> PrintLocations;

// 扩展的PragmaLearnHandler - 存储位置信息
struct AdvancedPragmaLearnHandler : public PragmaHandler {
    AdvancedPragmaLearnHandler() : PragmaHandler("learn") {}
    
    void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                      Token &FirstToken) override {
        SourceLocation PragmaLoc = FirstToken.getLocation();
        
        Token Tok;
        PP.Lex(Tok);
        
        if (Tok.isNot(tok::identifier)) {
            return;
        }
        
        IdentifierInfo *Directive = Tok.getIdentifierInfo();
        PP.Lex(Tok);
        
        if (Directive->isStr("print")) {
            // 存储位置信息供后续使用
            LearnPrintLocation LocInfo;
            LocInfo.Loc = PragmaLoc;
            LocInfo.Message = "Print statement inserted by #pragma learn";
            PrintLocations.push_back(LocInfo);
        }
    }
};

// AST Visitor - 遍历AST并插入printf
class LearnASTVisitor : public RecursiveASTVisitor<LearnASTVisitor> {
public:
    explicit LearnASTVisitor(ASTContext &Context) : Context(Context) {}
    
    bool VisitStmt(Stmt *S) {
        // 检查语句位置是否匹配我们存储的#pragma learn print位置
        for (const auto &LocInfo : PrintLocations) {
            if (Context.getSourceManager().isBeforeInTranslationUnit(
                    LocInfo.Loc, S->getBeginLoc())) {
                // 在这里插入printf调用
                llvm::errs() << "Would insert printf at: " 
                             << LocInfo.Loc.printToString(Context.getSourceManager()) 
                             << "\n";
            }
        }
        return true;
    }
    
private:
    ASTContext &Context;
};

// AST Consumer
class LearnASTConsumer : public ASTConsumer {
public:
    explicit LearnASTConsumer(ASTContext &Context) : Visitor(Context) {}
    
    void HandleTranslationUnit(ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
    
private:
    LearnASTVisitor Visitor;
};

// Frontend Action
class LearnFrontendAction : public ASTFrontendAction {
public:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(
        CompilerInstance &CI, StringRef InFile) override {
        return std::make_unique<LearnASTConsumer>(CI.getASTContext());
    }
};

// ============================================
// 示例2：支持更多指令和参数
// ============================================

struct EnhancedPragmaLearnHandler : public PragmaHandler {
    EnhancedPragmaLearnHandler() : PragmaHandler("learn") {}
    
    void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                      Token &FirstToken) override {
        SourceLocation PragmaLoc = FirstToken.getLocation();
        
        Token Tok;
        PP.Lex(Tok);
        
        if (Tok.is(tok::eod)) {
            PP.Diag(PragmaLoc, diag::err_pragma_missing_argument)
                << "learn" << true << "identifier";
            return;
        }
        
        if (Tok.isNot(tok::identifier)) {
            PP.Diag(Tok.getLocation(), diag::err_pragma_learn_unknown_directive)
                << Tok.getName();
            return;
        }
        
        IdentifierInfo *Directive = Tok.getIdentifierInfo();
        PP.Lex(Tok);
        
        // 处理带参数的指令
        if (Directive->isStr("warn")) {
            handleWarnDirective(PP, PragmaLoc, Tok);
        } else if (Directive->isStr("print")) {
            handlePrintDirective(PP, PragmaLoc, Tok);
        } else if (Directive->isStr("info")) {
            handleInfoDirective(PP, PragmaLoc, Tok);
        } else if (Directive->isStr("trace")) {
            handleTraceDirective(PP, PragmaLoc, Tok);
        } else if (Directive->isStr("assert")) {
            handleAssertDirective(PP, PragmaLoc, Tok);
        } else {
            PP.Diag(PragmaLoc, diag::err_pragma_learn_unknown_directive)
                << Directive->getName();
            PP.Diag(PragmaLoc, diag::note_pragma_learn_available_directives);
        }
    }
    
private:
    void handleWarnDirective(Preprocessor &PP, SourceLocation Loc, Token &Tok) {
        // 支持自定义警告消息
        std::string Message = "Custom warning from #pragma learn";
        
        if (Tok.is(tok::string_literal)) {
            Message = PP.getSpelling(Tok);
            // 去掉引号
            if (Message.size() >= 2) {
                Message = Message.substr(1, Message.size() - 2);
            }
            PP.Lex(Tok);
        }
        
        PP.Diag(Loc, diag::warn_pragma_learn_warn_directive) << Message;
    }
    
    void handlePrintDirective(Preprocessor &PP, SourceLocation Loc, Token &Tok) {
        // 支持printf格式字符串
        std::string Format = "Hello from #pragma learn";
        
        if (Tok.is(tok::string_literal)) {
            Format = PP.getSpelling(Tok);
            if (Format.size() >= 2) {
                Format = Format.substr(1, Format.size() - 2);
            }
            PP.Lex(Tok);
        }
        
        // 存储格式字符串供CodeGen使用
        PP.Diag(Loc, diag::remark_pragma_learn_print) << Format;
    }
    
    void handleInfoDirective(Preprocessor &PP, SourceLocation Loc, Token &Tok) {
        // 发出信息性备注
        PP.Diag(Loc, diag::remark_pragma_learn_print) 
            << "Info message from #pragma learn";
    }
    
    void handleTraceDirective(Preprocessor &PP, SourceLocation Loc, Token &Tok) {
        // 跟踪指令 - 可以用于函数进入/退出跟踪
        PP.Diag(Loc, diag::remark_pragma_learn_print)
            << "Trace point from #pragma learn";
    }
    
    void handleAssertDirective(Preprocessor &PP, SourceLocation Loc, Token &Tok) {
        // 断言指令
        PP.Diag(Loc, diag::warn_pragma_learn_warn_directive)
            << "Assertion point from #pragma learn";
    }
};

// ============================================
// 示例3：支持作用域（push/pop）
// ============================================

struct ScopedPragmaLearnHandler : public PragmaHandler {
    ScopedPragmaLearnHandler() : PragmaHandler("learn") {}
    
    void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                      Token &FirstToken) override {
        SourceLocation PragmaLoc = FirstToken.getLocation();
        
        Token Tok;
        PP.Lex(Tok);
        
        if (Tok.isNot(tok::identifier)) {
            return;
        }
        
        IdentifierInfo *Directive = Tok.getIdentifierInfo();
        PP.Lex(Tok);
        
        if (Directive->isStr("push")) {
            // 保存当前状态
            StateStack.push(CurrentState);
            PP.Diag(PragmaLoc, diag::remark_pragma_learn_print)
                << "Pushed learn pragma state";
        } else if (Directive->isStr("pop")) {
            // 恢复之前的状态
            if (!StateStack.empty()) {
                CurrentState = StateStack.top();
                StateStack.pop();
                PP.Diag(PragmaLoc, diag::remark_pragma_learn_print)
                    << "Popped learn pragma state";
            } else {
                PP.Diag(PragmaLoc, diag::warn_pragma_learn_warn_directive)
                    << "No state to pop";
            }
        } else if (Directive->isStr("warn")) {
            CurrentState.WarnEnabled = true;
            PP.Diag(PragmaLoc, diag::warn_pragma_learn_warn_directive);
        } else if (Directive->isStr("nowarn")) {
            CurrentState.WarnEnabled = false;
            PP.Diag(PragmaLoc, diag::remark_pragma_learn_print)
                << "Warnings disabled";
        }
    }
    
private:
    struct LearnState {
        bool WarnEnabled = false;
        bool PrintEnabled = false;
        bool TraceEnabled = false;
    };
    
    LearnState CurrentState;
    std::stack<LearnState> StateStack;
};

// ============================================
// 示例4：使用注解token传递信息给Parser
// ============================================

// 首先需要在TokenKinds.def中添加新的token类型
// def annot_pragma_learn : Annotation;

struct AnnotatingPragmaLearnHandler : public PragmaHandler {
    AnnotatingPragmaLearnHandler() : PragmaHandler("learn") {}
    
    void HandlePragma(Preprocessor &PP, PragmaIntroducer Introducer,
                      Token &FirstToken) override {
        SourceLocation StartLoc = FirstToken.getLocation();
        
        Token Tok;
        PP.Lex(Tok);
        
        if (Tok.isNot(tok::identifier)) {
            return;
        }
        
        IdentifierInfo *Directive = Tok.getIdentifierInfo();
        PP.Lex(Tok);
        
        // 确定指令类型
        enum LearnDirectiveKind {
            LDK_Warn,
            LDK_Print,
            LDK_Info,
            LDK_Unknown
        };
        
        LearnDirectiveKind Kind = LDK_Unknown;
        if (Directive->isStr("warn")) Kind = LDK_Warn;
        else if (Directive->isStr("print")) Kind = LDK_Print;
        else if (Directive->isStr("info")) Kind = LDK_Info;
        
        // 跳过直到行尾
        while (Tok.isNot(tok::eod)) {
            PP.Lex(Tok);
        }
        
        SourceLocation EndLoc = Tok.getLocation();
        
        // 创建注解token
        MutableArrayRef<Token> Toks(
            PP.getPreprocessorAllocator().Allocate<Token>(1), 1);
        
        Toks[0].startToken();
        Toks[0].setKind(tok::annot_pragma_learn); // 需要定义这个token类型
        Toks[0].setLocation(StartLoc);
        Toks[0].setAnnotationEndLoc(EndLoc);
        
        // 存储指令类型作为注解值
        Toks[0].setAnnotationValue(
            reinterpret_cast<void*>(static_cast<uintptr_t>(Kind)));
        
        // 将注解token注入到token流中
        PP.EnterTokenStream(Toks, /*DisableMacroExpansion=*/true,
                            /*IsReinject=*/false);
    }
};

// 然后在Parser中添加处理函数
// void Parser::HandlePragmaLearn() {
//     assert(Tok.is(tok::annot_pragma_learn));
//     LearnDirectiveKind Kind = static_cast<LearnDirectiveKind>(
//         reinterpret_cast<uintptr_t>(Tok.getAnnotationValue()));
//     SourceLocation Loc = ConsumeAnnotationToken();
//     
//     // 根据Kind执行相应操作
//     switch (Kind) {
//         case LDK_Warn:
//             // 处理warn指令
//             break;
//         case LDK_Print:
//             // 处理print指令
//             break;
//         // ...
//     }
// }

// ============================================
// 总结
// ============================================

/*
这些示例展示了多种扩展#pragma learn的方式：

1. 基础实现 - 当前已实现的版本
2. AST Consumer - 用于真正的代码插入
3. 增强指令 - 支持参数和更多指令类型
4. 作用域管理 - push/pop状态
5. 注解token - 通过token流与Parser通信

选择哪种方式取决于具体需求：
- 简单诊断 -> 基础实现
- 代码插入 -> AST Consumer + CodeGen
- 复杂解析 -> 注解token + Parser处理
- 状态管理 -> push/pop作用域
*/
