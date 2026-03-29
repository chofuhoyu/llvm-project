//===-- LearnPragmaTool.cpp - Tool to process #pragma learn print -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool processes source files containing #pragma learn print directives
// and automatically inserts printf statements and necessary includes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <regex>
#include <string>

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

// Options
static cl::OptionCategory LearnPragmaCategory("learn-pragma options");
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);
static cl::extrahelp MoreHelp(
    "\nThis tool processes #pragma learn print directives in source files.\n"
    "It inserts printf statements and adds necessary #include directives.\n");

namespace {

// Structure to hold information about a #pragma learn print location
struct PragmaLearnInfo {
  SourceLocation Location;
  std::string Identifier;
  bool HasIdentifier;
};

// AST Visitor to find and process #pragma learn print directives
class LearnPragmaVisitor : public RecursiveASTVisitor<LearnPragmaVisitor> {
public:
  explicit LearnPragmaVisitor(Rewriter &R, SourceManager &SM)
      : TheRewriter(R), SM(SM) {}

  bool VisitPragmaComment(PragmaCommentDecl *D) {
    StringRef Comment = D->getComment();

    // Check if this is a #pragma learn print
    if (Comment.startswith("learn print")) {
      PragmaLearnInfo Info;
      Info.Location = D->getLocation();

      // Parse the identifier if present
      std::string CommentStr = Comment.str();
      std::regex PrintRegex(R"(learn\s+print\s+(\w+))");
      std::smatch Match;

      if (std::regex_search(CommentStr, Match, PrintRegex) &&
          Match.size() > 1) {
        Info.Identifier = Match[1].str();
        Info.HasIdentifier = true;
      } else {
        Info.HasIdentifier = false;
      }

      PragmaLocations.push_back(Info);
    }
    return true;
  }

  const std::vector<PragmaLearnInfo> &getPragmaLocations() const {
    return PragmaLocations;
  }

private:
  Rewriter &TheRewriter;
  SourceManager &SM;
  std::vector<PragmaLearnInfo> PragmaLocations;
};

// AST Consumer that uses our visitor
class LearnPragmaASTConsumer : public ASTConsumer {
public:
  explicit LearnPragmaASTConsumer(Rewriter &R, SourceManager &SM)
      : Visitor(R, SM) {}

  void HandleTranslationUnit(ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }

  const LearnPragmaVisitor &getVisitor() const { return Visitor; }

private:
  LearnPragmaVisitor Visitor;
};

// Frontend Action
class LearnPragmaFrontendAction : public ASTFrontendAction {
public:
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<LearnPragmaASTConsumer>(TheRewriter,
                                                    CI.getSourceManager());
  }

  void EndSourceFileAction() override {
    // Get the consumer and visitor
    LearnPragmaASTConsumer *Consumer =
        static_cast<LearnPragmaASTConsumer *>(getCurrentASTConsumer().get());
    const LearnPragmaVisitor &Visitor = Consumer->getVisitor();

    SourceManager &SM = TheRewriter.getSourceMgr();
    FileID MainFileID = SM.getMainFileID();

    // Check if we need to add stdio.h include
    bool NeedsStdioInclude = !Visitor.getPragmaLocations().empty();
    bool HasStdioInclude = false;

    if (NeedsStdioInclude) {
      // Scan the file for existing stdio.h or cstdio includes
      bool Invalid = false;
      StringRef FileContent = SM.getBufferData(MainFileID, &Invalid);

      if (!Invalid) {
        std::string Content = FileContent.str();
        if (Content.find("#include <stdio.h>") != std::string::npos ||
            Content.find("#include \"stdio.h\"") != std::string::npos ||
            Content.find("#include <cstdio>") != std::string::npos ||
            Content.find("#include \"cstdio\"") != std::string::npos) {
          HasStdioInclude = true;
        }
      }
    }

    // Add stdio.h include if needed
    if (NeedsStdioInclude && !HasStdioInclude) {
      // Determine if it's C++
      StringRef Filename = SM.getFileEntryForID(MainFileID)->getName();
      bool IsCPP = Filename.endswith(".cpp") || Filename.endswith(".cxx") ||
                   Filename.endswith(".cc") || Filename.endswith(".C");

      std::string IncludeStmt =
          IsCPP ? "#include <cstdio>\n" : "#include <stdio.h>\n";

      // Insert at the beginning of the file
      SourceLocation FileStart = SM.getLocForStartOfFile(MainFileID);
      TheRewriter.InsertText(FileStart, IncludeStmt);
      llvm::outs() << "Added: " << IncludeStmt;
    }

    // Insert printf statements for each pragma
    for (const auto &Info : Visitor.getPragmaLocations()) {
      // Find the end of the line containing the pragma
      SourceLocation InsertLoc = Info.Location;
      bool Invalid = false;

      // Move to the end of the line
      while (true) {
        const char *Ptr = SM.getCharacterData(InsertLoc, &Invalid);
        if (Invalid || !*Ptr)
          break;
        if (*Ptr == '\n' || *Ptr == '\r')
          break;
        InsertLoc = InsertLoc.getLocWithOffset(1);
      }

      // Move to beginning of next line
      if (!Invalid) {
        InsertLoc = InsertLoc.getLocWithOffset(1);
      }

      // Build the printf statement
      std::string PrintfStmt;
      if (Info.HasIdentifier) {
        PrintfStmt =
            "    printf(\"#pragma learn print directive received identifier " +
            Info.Identifier + "\\n\");\n";
      } else {
        PrintfStmt =
            "    printf(\"#pragma learn print directive received\\n\");\n";
      }

      // Insert the printf statement
      TheRewriter.InsertText(InsertLoc, PrintfStmt);
      llvm::outs() << "Inserted at line "
                   << SM.getSpellingLineNumber(Info.Location) << ": "
                   << PrintfStmt;
    }

    // Write the modified file back
    const RewriteBuffer *RewriteBuf =
        TheRewriter.getRewriteBufferFor(MainFileID);
    if (RewriteBuf) {
      std::string OutputFilename =
          SM.getFileEntryForID(MainFileID)->getName().str();

      // Write to a temporary file first, then replace
      std::string TempFilename = OutputFilename + ".tmp";
      std::error_code EC;
      llvm::raw_fd_ostream OS(TempFilename, EC, llvm::sys::fs::OF_Text);

      if (!EC) {
        RewriteBuf->write(OS);
        OS.close();

        // Replace the original file
        if (llvm::sys::fs::rename(TempFilename, OutputFilename)) {
          llvm::errs() << "Error replacing file: " << OutputFilename << "\n";
        } else {
          llvm::outs() << "\nSuccessfully modified: " << OutputFilename << "\n";
        }
      } else {
        llvm::errs() << "Error opening temp file: " << EC.message() << "\n";
      }
    } else {
      llvm::outs() << "No changes needed for this file.\n";
    }
  }

private:
  Rewriter TheRewriter;
};

// Factory for our action
class LearnPragmaFrontendActionFactory : public FrontendActionFactory {
public:
  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<LearnPragmaFrontendAction>();
  }
};

} // anonymous namespace

int main(int argc, const char **argv) {
  auto ExpectedParser =
      CommonOptionsParser::create(argc, argv, LearnPragmaCategory);
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  CommonOptionsParser &OptionsParser = ExpectedParser.get();
  ClangTool Tool(OptionsParser.getCompilations(),
                 OptionsParser.getSourcePathList());

  llvm::outs() << "Processing #pragma learn print directives...\n\n";

  return Tool.run(new LearnPragmaFrontendActionFactory());
}
